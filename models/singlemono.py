import pytorch_lightning as pl
import torch
import torch.nn as nn

from loss.lovasz_losses import lovasz_softmax
from loss.sscMetrics import SSCMetrics
from loss.ssc_loss import global_loss, CE_ssc_loss, CE_semantic_loss
from models.sflosp import FLoSP
import numpy as np

from models.sunet3d_kitti import UNet3DDecoder
from models.sunet2d_feature import UNet2D
from torch.optim.lr_scheduler import MultiStepLR


class SingleMono(pl.LightningModule):
    def __init__(
            self,
            n_classes,
            class_names,
            feature,
            class_weights, seg_class_weights,
            project_scale,
            full_scene_size,
            dataset,
            context_prior=True,
            fp_loss=True,
            frustum_size=4,
            relation_loss=False,
            CE_ssc_loss=True,
            geo_scal_loss=True,
            lr=1e-4,
            weight_decay=1e-4,
    ):
        super().__init__()

        self.fp_loss = fp_loss
        self.dataset = dataset
        self.context_prior = context_prior
        self.frustum_size = frustum_size
        self.class_names = class_names
        self.relation_loss = relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.geo_scal_loss = geo_scal_loss
        self.project_scale = project_scale
        self.class_weights = class_weights
        self.seg_class_weights = seg_class_weights
        self.lr = lr
        self.weight_decay = weight_decay

        self.projects_encoder = {}
        self.scale_3ds = [32, 16, 8, 4, 2, 1]  # 2D scales
        for scale_3d in self.scale_3ds:
            self.projects_encoder[str(scale_3d)] = FLoSP(
                full_scene_size, project_scale=scale_3d, dataset=self.dataset
            )
        self.projects_encoder = nn.ModuleDict(self.projects_encoder)

        self.n_classes = n_classes

        self.net_3d_decoder = UNet3DDecoder(
            self.n_classes,
            nn.BatchNorm3d,
            project_scale=project_scale,
            feature=feature,
            full_scene_size=full_scene_size,
            context_prior=context_prior,
        )
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

        self.save_hyperparameters()
        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)

    def forward(self, batch):
        torch.use_deterministic_algorithms(False)
        img = batch["img"]
        bs = len(img)

        out = {}
        x_rgb, seg_out, reg_out = self.net_rgb(img)
        x_rgb_encoder = x_rgb

        x3ds_encoder = []
        for scale_3d in self.scale_3ds:
            for i in range(bs):
                scale_3d = int(scale_3d)
                projected_pix = batch["projected_pix_{}".format(scale_3d)][i].cuda()
                fov_mask = batch["fov_mask_{}".format(scale_3d)][i].cuda()
                x3d = self.projects_encoder[str(scale_3d)](
                    x_rgb_encoder["1_" + str(scale_3d)][i],
                    projected_pix // scale_3d,
                    fov_mask,
                )
                if i == 0:
                    temp_s = x3d.unsqueeze(dim=0)
                else:
                    temp_s = torch.cat([temp_s, x3d.unsqueeze(dim=0)], 0)
            x3ds_encoder.append(temp_s)

        out = self.net_3d_decoder(x3ds_encoder)
        out["sem_logit"] = seg_out
        out["sem_depth"] = reg_out
        return out

    def step(self, batch, step_type, metric):

        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]

        seg_class_weight = self.seg_class_weights.type_as(batch["img"])
        sem_logit = out_dict["sem_logit"]

        proj_label = batch["proj_label"].permute(0, 2, 1)
        loss_func = torch.nn.CrossEntropyLoss(weight=seg_class_weight, ignore_index=0)
        sem_ce_loss = loss_func(sem_logit, proj_label.long())
        sem_lovasz_loss = lovasz_softmax(torch.nn.functional.softmax(sem_logit, dim=1),
                                         proj_label.contiguous().long(), ignore=0)


        loss += sem_ce_loss
        loss += sem_lovasz_loss
        self.log(
            step_type + "/sem_ce_loss",
            sem_ce_loss.detach(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            step_type + "/sem_lovasz_loss",
            sem_lovasz_loss.detach(),
            on_epoch=True,
            sync_dist=True,
        )

        sem_depth = out_dict["sem_depth"]

        proj_depth = 0.1 * batch["proj_depth"].permute(0, 2, 1)
        mask = (proj_depth >= 0.1) * (proj_depth <= 6.4)
        mask.detach_()
        sem_depth_loss = torch.nn.functional.smooth_l1_loss(sem_depth[mask], proj_depth[mask], size_average=True)
        loss += sem_depth_loss
        self.log(
            step_type + "/sem_depth_loss",
            sem_depth_loss.detach(),
            on_epoch=True,
            sync_dist=True,
        )
        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        '''semantic loss'''
        ssc_semantic = CE_semantic_loss(ssc_pred, target, class_weight)
        loss += ssc_semantic
        self.log(
            step_type + "/ssc_semantic",
            ssc_semantic.detach(),
            on_epoch=True,
            sync_dist=True,
        )

        loss_sem_scal = global_loss(ssc_pred, target)
        loss += loss_sem_scal
        self.log(
            step_type + "/loss_global",
            loss_sem_scal.detach(),
            on_epoch=True,
            sync_dist=True,
        )

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()

    def configure_optimizers(self):
        if self.dataset == "NYU":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.dataset == "kitti":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            # scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25, 30, 35], gamma=0.7)
            return [optimizer], [scheduler]


