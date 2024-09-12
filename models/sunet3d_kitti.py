# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
#from monoscene.models.modules import SegmentationHead
from models.CRP3D import CPMegaVoxels
#from monoscene.models.modules import Process, Upsample, Downsample
from models.modules import Process, Downsample
def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(True)
    )
class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        #out_channels = int(in_channels / 4)
        out_channels = int(in_channels / 2)
        self.avgpool1 = nn.AdaptiveAvgPool3d((128, 128, 16))
        self.avgpool2 = nn.AdaptiveAvgPool3d((64, 64, 8))
        self.avgpool3 = nn.AdaptiveAvgPool3d((32, 32, 32))
        self.avgpool4 = nn.AdaptiveAvgPool3d((16, 16, 32))
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.maxpool1 = nn.AdaptiveMaxPool3d((128, 128, 16))
        self.maxpool2 = nn.AdaptiveMaxPool3d((64, 64, 8))

    def forward(self, x):
        size = x.size()[2:]
        #print(size)
        '''feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='trilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='trilinear', align_corners=True)'''
        feat1 = F.interpolate(self.conv1(self.maxpool1(x)), size, mode='trilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.maxpool2(x)), size, mode='trilinear', align_corners=True)
        '''feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='trilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='trilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)'''
        return torch.cat([x, feat1, feat2], dim=1)
class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        feature,
        project_scale,
        context_prior=None,
        bn_momentum=0.1,
    ):
        super(UNet3D, self).__init__()
        self.business_layer = []
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature

        size_l1 = (
            int(self.full_scene_size[0] / project_scale),
            int(self.full_scene_size[1] / project_scale),
            int(self.full_scene_size[2] / project_scale),
        )
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        dilations = [1, 2, 3]
        self.process_l1 = nn.Sequential(
            Process(self.feature, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Process(self.feature * 2, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature * 2, norm_layer, bn_momentum),
        )

        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_12_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        self.up_l1_lfull = Upsample(
            self.feature, self.feature // 2, norm_layer, bn_momentum
        )

        self.ssc_head = SegmentationHead(
            self.feature // 2, self.feature // 2, class_num, dilations
        )

        self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature * 4, size_l3, bn_momentum=bn_momentum
            )

    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"]

        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l3 = self.process_l2(x3d_l2)

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_l3)
            x3d_l3 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        ssc_logit_full = self.ssc_head(x3d_up_lfull)

        res["ssc_logit"] = ssc_logit_full

        return res
class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    Taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        
        self.drop3d_1m = nn.Dropout3d(p=0.3)
        self.drop3d_2m = nn.Dropout3d(p=0.3)
        self.drop3d_3m = nn.Dropout3d(p=0.3)
        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

        self.conv_classes = nn.Conv3d(
            planes, nbr_classes, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x_in):

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.drop3d_1m(self.conv0(x_in)))

        y = self.bn2[0](self.drop3d_2m(self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in))))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.drop3d_3m(self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in))))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in

class UNet3DDecoder(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        feature,
        project_scale,
        context_prior=None,
        bn_momentum=0.1,
    ):
        super(UNet3DDecoder, self).__init__()
        self.business_layer = []
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature

        self.up_13_l16 = Upsample(
            self.feature * 32, self.feature * 16, norm_layer, bn_momentum
        )
        self.up_13_l8 = Upsample(
            self.feature * 16, self.feature * 8, norm_layer, bn_momentum
        )
        self.up_13_l4 = Upsample(
            self.feature * 8, self.feature * 4, norm_layer, bn_momentum
        )
        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_13_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )

        self.up_l1_lfull = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        self.psp = _PyramidPooling(self.feature, norm_layer=norm_layer, norm_kwargs=None)
        self.myconv = nn.Conv3d(
            self.feature * 2,
            self.feature,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.mynorm = norm_layer(self.feature, momentum=bn_momentum)
        self.myrelu = nn.ReLU(True)
        self.myconv1 = nn.Conv3d(
            self.feature * 2,
            self.feature,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.mynorm1 = norm_layer(self.feature, momentum=bn_momentum)
        self.myrelu1 = nn.ReLU(True)
        self.mydrop = nn.Dropout3d(0.3)
        self.conv_classes = nn.Conv3d(
            self.feature, class_num, kernel_size=3, padding=1, stride=1
        )

    def forward(self, input_dict):
        res = {}
        x3d_l32 = input_dict[0]
        x3d_l16 = input_dict[1]
        x3d_l8 = input_dict[2]
        x3d_l4 = input_dict[3]
        x3d_l2 = input_dict[4]
        x3d_l1 = input_dict[5]
        x3d_up_l16 = self.up_13_l16(x3d_l32) + x3d_l16
        x3d_up_l8 = self.up_13_l8(x3d_up_l16) + x3d_l8
        x3d_up_l4 = self.up_13_l4(x3d_up_l8) + x3d_l4
        x3d_up_l2 = self.up_13_l2(x3d_up_l4) + x3d_l2
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l2)

        x3d_up_lfull = self.psp(x3d_up_lfull)
        x3d_up_lfull = self.mydrop(self.myrelu(self.mynorm(self.myconv(x3d_up_lfull))))
        x3d_up_lfull = torch.cat([x3d_up_lfull, x3d_l1], dim=1)
        x3d_up_lfull = self.myrelu1(self.mynorm1(self.myconv1(x3d_up_lfull)))
        ssc_logit_full = self.conv_classes(x3d_up_lfull)
        res["ssc_logit"] = ssc_logit_full

        return res
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)
class UNet3DDecoder_salax(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        feature,
        project_scale,
        context_prior=None,
        bn_momentum=0.1,
    ):
        super(UNet3DDecoder_salax, self).__init__()
        self.business_layer = []
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature

        size_l1 = (
            int(self.full_scene_size[0] / project_scale),
            int(self.full_scene_size[1] / project_scale),
            int(self.full_scene_size[2] / project_scale),
        )
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        dilations = [1, 2, 3]
        self.up_13_l16 = Upsample(
            self.feature * 32, self.feature * 16, norm_layer, bn_momentum
        )
        self.up_13_l8 = Upsample(
            self.feature * 8, self.feature * 4, norm_layer, bn_momentum
        )
        self.up_13_l4 = Upsample(
            self.feature * 4, self.feature * 4, norm_layer, bn_momentum
        )

        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_13_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        '''
        self.up_l1_lfull = Upsample(
            self.feature, self.feature // 2, norm_layer, bn_momentum
        )
        self.ssc_head = SegmentationHead(
            self.feature // 2, self.feature // 2, class_num, dilations
        )
        '''
        self.up_l1_lfull = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        self.myconv = nn.Conv3d(
            self.feature,
            self.feature,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.mynorm = norm_layer(self.feature, momentum=bn_momentum)
        self.myrelu = nn.ReLU()
        '''self.ssc_head = SegmentationHead(
            self.feature // 2, self.feature // 2, class_num, dilations
        )
        self.myconv1 = nn.Conv3d(
            self.feature // 2,
            self.feature // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.mynorm1 = norm_layer(self.feature // 2, momentum=bn_momentum)
        self.myrelu1 = nn.ReLU()'''
        self.conv_classes = nn.Conv3d(
            self.feature, class_num, kernel_size=3, padding=1, stride=1
        )

    def forward(self, input_dict):
        res = {}
        x3d_l16 = input_dict[0]
        x3d_l8 = input_dict[1]
        x3d_l4 = input_dict[2]
        x3d_l2 = input_dict[3]
        x3d_l1 = input_dict[4]
        x3d_up_l8 = self.up_13_l8(x3d_l16) + x3d_l8

        x3d_up_l4 = self.up_13_l4(x3d_up_l8) + x3d_l4
        x3d_up_l2 = self.up_13_l2(x3d_up_l4) + x3d_l2
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.myrelu(self.mynorm(self.myconv(x3d_up_lfull)))
        #ssc_logit_full = self.ssc_head(x3d_up_lfull)
        ssc_logit_full = self.conv_classes(x3d_up_lfull)
        res["ssc_logit"] = ssc_logit_full

        return res
