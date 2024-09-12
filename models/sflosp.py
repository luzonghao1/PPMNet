import math

import torch
import torch.nn as nn
import torch.nn.functional as F
def _PSP1x1Conv(in_channels, out_channels, norm_layer):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class FLoSP(nn.Module):
    def __init__(self, scene_size, dataset, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale

    def forward(self, x2d, projected_pix, fov_mask):
        c, h, w = x2d.shape
        src = x2d.view(c, -1)
        zeros_vec = torch.zeros(c, 1).type_as(src)
        src = torch.cat([src, zeros_vec], 1)

        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
        img_indices = pix_y * w + pix_x
        img_indices[~fov_mask] = h * w
        img_indices = img_indices.expand(c, -1).long()  # c, HWD
        src_feature = torch.gather(src, 1, img_indices)
        if self.dataset == "NYU":
            x3d = src_feature.reshape(
                c,
                math.ceil(self.scene_size[0] / self.project_scale),
                math.ceil(self.scene_size[2] / self.project_scale),
                math.ceil(self.scene_size[1] / self.project_scale),
            )

            '''x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                self.scene_size[1] // self.project_scale,
            )'''
            x3d = x3d.permute(0, 1, 3, 2)
        elif self.dataset == "kitti":
            x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                self.scene_size[2] // self.project_scale,
            )

        return x3d
class FLoSPin(nn.Module):
    def __init__(self, scene_size, dataset, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale

    def forward(self, x2d, projected_pix, fov_mask):
        c, h, w = x2d.shape
        self.convNon1 = _PSP1x1Conv(c, c, norm_layer=nn.BatchNorm2d).cuda()
        x2d = x2d.unsqueeze(dim=0)
        x2d = self.convNon1(x2d).unsqueeze(dim=4)

        if self.dataset == "NYU":
            x3d = F.interpolate(x2d, (math.ceil(self.scene_size[0] / self.project_scale),
                                              math.ceil(self.scene_size[2] / self.project_scale),
                                              math.ceil(self.scene_size[1] / self.project_scale),), mode='trilinear',
                                        align_corners=True).squeeze(dim=0)
            x3d = x3d.permute(0, 1, 3, 2)
        elif self.dataset == "kitti":
            x3d = F.interpolate(x2d, (self.scene_size[0] / self.project_scale,
                                              self.scene_size[2] / self.project_scale,
                                              self.scene_size[1] / self.project_scale,), mode='trilinear',
                                        align_corners=True).squeeze(dim=0)
        return x3d
