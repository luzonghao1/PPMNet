from data.semantic_kitti.skitti_dm import KittiDataModule
from data.semantic_kitti.params import (
    semantic_kitti_class_frequencies,
    kitti_class_names, seg_num_per_class,
)

from models.singlemono import SingleMono
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch

hydra.output_subdir = None


@hydra.main(config_name="../config/ppmnet.yaml")
def main(config: DictConfig):
    exp_name = "ppmnet"

    class_names = kitti_class_names
    max_epochs = 50
    logdir = config.kitti_logdir
    full_scene_size = (256, 256, 32)
    project_scale = 2
    feature = 64
    n_classes = 20
    class_weights = torch.from_numpy(
        1 / np.log(semantic_kitti_class_frequencies + 0.001)
    )
    seg_class_weights = torch.from_numpy(
        1 / np.log(seg_num_per_class + 0.001)
    )
    data_module = KittiDataModule(
        root=config.kitti_root,
        preprocess_root=config.kitti_preprocess_root,
        frustum_size=config.frustum_size,
        project_scale=project_scale,
        batch_size=int(config.batch_size / config.n_gpus),
        num_workers=int(config.num_workers_per_gpu),
    )

    # Initialize MonoScene model
    model = SingleMono(
        dataset=config.dataset,
        frustum_size=config.frustum_size,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        feature=feature,
        full_scene_size=full_scene_size,
        n_classes=n_classes,
        class_names=class_names,
        context_prior=config.context_prior,
        relation_loss=config.relation_loss,
        CE_ssc_loss=config.CE_ssc_loss,
        sem_scal_loss=config.sem_scal_loss,
        geo_scal_loss=config.geo_scal_loss,
        lr=config.lr,
        weight_decay=config.weight_decay,
        class_weights=class_weights,
        seg_class_weights=seg_class_weights,
    )

    if config.enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val/mIoU",
                save_top_k=1,
                mode="max",
                filename="{epoch:03d}-{val/mIoU:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=True,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=True,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

