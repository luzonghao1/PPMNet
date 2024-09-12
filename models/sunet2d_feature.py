"""
Code adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.ReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)
class UpSampleBN_feature(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN_feature, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.ReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class DecoderBN(nn.Module):
    def __init__(
        self, num_features, bottleneck_features, out_feature, use_decoder=True
    ):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder
        self.feature_1_16 = 2048 // 2
        self.feature_1_8 = 2048 // 4
        self.feature_1_4 = 2048 // 8
        self.feature_1_2 = 2048 // 16
        self.feature_1_1 = 2048 // 32 #32
        self.conv2 = nn.Conv2d(
            bottleneck_features, 2048, kernel_size=1, stride=1, padding=1
        )
        self.up16 = UpSampleBN(
            skip_input=2048 + 224, output_features=self.feature_1_16
        )
        self.up8 = UpSampleBN(
            skip_input=self.feature_1_16 + 80, output_features=self.feature_1_8
        )
        self.up4 = UpSampleBN(
            skip_input=self.feature_1_8 + 48, output_features=self.feature_1_4
        )
        self.up2 = UpSampleBN(
            skip_input=self.feature_1_4 + 32, output_features=self.feature_1_2
        )
        self.up1 = UpSampleBN(
            skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1
        )

        self.up16_feature = UpSampleBN_feature(
            skip_input=2048 + 224, output_features=self.feature_1_16
        )
        self.up8_feature = UpSampleBN_feature(
            skip_input=self.feature_1_16 + 80, output_features=self.feature_1_8
        )
        self.up4_feature = UpSampleBN_feature(
            skip_input=self.feature_1_8 + 48, output_features=self.feature_1_4
        )
        self.up2_feature = UpSampleBN_feature(
            skip_input=self.feature_1_4 + 32, output_features=self.feature_1_2
        )
        self.up1_feature = UpSampleBN_feature(
            skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1
        )


    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )

        x_d0 = self.conv2(x_block4)
        x_1_16 = self.up16(x_d0, x_block3)
        x_1_8 = self.up8(x_1_16, x_block2)
        x_1_4 = self.up4(x_1_8, x_block1)
        x_1_2 = self.up2(x_1_4, x_block0)
        x_1_1 = self.up1(x_1_2, features[0])

        return {
            "1_1": x_1_1,
            "1_2": x_1_2,
            "1_4": x_1_4,
            "1_8": x_1_8,
            "1_16": x_1_16,
            "1_32": x_d0,
            "2_1": x_1_1
        }

class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class _PyramidPooling2d_kitti(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling2d_kitti, self).__init__()
        out_channels = int(in_channels / 2)
        self.avgpool1 = nn.AdaptiveAvgPool2d((185, 610))
        self.avgpool2 = nn.AdaptiveAvgPool2d((93, 305))
        self.maxpool1 = nn.AdaptiveMaxPool2d((185, 610))
        self.maxpool2 = nn.AdaptiveMaxPool2d((93, 305))

        self.conv1 = _PSP1x1Conv2d(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv2d(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv2d(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.maxpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.maxpool2(x)), size, mode='bilinear', align_corners=True)

        return torch.cat([x, feat1, feat2], dim=1)
class UNet2D(nn.Module):
    def __init__(self, backend, num_features, out_feature, use_decoder=True):
        super(UNet2D, self).__init__()
        self.use_decoder = use_decoder
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN(
            out_feature=out_feature,
            use_decoder=use_decoder,
            bottleneck_features=num_features,
            num_features=num_features,
        )
        self.psp = _PyramidPooling2d_kitti(64, norm_layer=nn.BatchNorm2d, norm_kwargs=None)

        self.seg_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.seg_bn2 = nn.BatchNorm2d(64)
        self.seg_relu2 = nn.ReLU(True)
        self.seg_drop = nn.Dropout(0.3)

        self.seg_conv2 = nn.Conv2d(64, 20, kernel_size=3, padding=1)

        self.reg_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x, **kwargs):
        encoded_feats = self.encoder(x)
        unet_out = self.decoder(encoded_feats, **kwargs)

        seg_out = self.psp(unet_out['1_1'])
        seg_out = self.seg_relu2(self.seg_bn2(self.seg_conv(seg_out)))
        seg_out = self.seg_drop(seg_out)
        unet_out['1_1'] = seg_out

        seg_out1 = self.seg_conv2(seg_out)

        reg_out = self.reg_conv2(seg_out)
        reg_out = F.softmax(reg_out, dim=1)
        reg_out = depthregression()(reg_out)
        return unet_out, seg_out1, reg_out * 0.1


    def get_encoder_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_decoder_params(self):  # lr learning rate
        return self.decoder.parameters()

    @classmethod
    def build(cls, **kwargs):
        basemodel_name = "tf_efficientnet_b7_ns"
        num_features = 2560

        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
        )
        print("Done.")

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, num_features=num_features, **kwargs)
        print("Done.")
        return m

class DecoderBN_NYU(nn.Module):
    def __init__(
        self, num_features, bottleneck_features, out_feature, use_decoder=True
    ):
        super(DecoderBN_NYU, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder
        self.feature_1_16 = 4096 // 2
        self.feature_1_8 = 4096 // 4
        self.feature_1_4 = 4096 // 8
        self.feature_1_2 = 4096 // 16
        self.feature_1_1 = 4096 // 32 #32
        self.conv2 = nn.Conv2d(
            bottleneck_features, 4096, kernel_size=1, stride=1, padding=1
        )
        self.up16 = UpSampleBN(
            skip_input=4096 + 224, output_features=self.feature_1_16
        )
        self.up8 = UpSampleBN(
            skip_input=self.feature_1_16 + 80, output_features=self.feature_1_8
        )
        self.up4 = UpSampleBN(
            skip_input=self.feature_1_8 + 48, output_features=self.feature_1_4
        )
        self.up2 = UpSampleBN(
            skip_input=self.feature_1_4 + 32, output_features=self.feature_1_2
        )
        self.up1 = UpSampleBN(
            skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1
        )

        self.up16_feature = UpSampleBN_feature(
            skip_input=4096 + 224, output_features=self.feature_1_16
        )
        self.up8_feature = UpSampleBN_feature(
            skip_input=self.feature_1_16 + 80, output_features=self.feature_1_8
        )
        self.up4_feature = UpSampleBN_feature(
            skip_input=self.feature_1_8 + 48, output_features=self.feature_1_4
        )
        self.up2_feature = UpSampleBN_feature(
            skip_input=self.feature_1_4 + 32, output_features=self.feature_1_2
        )
        self.up1_feature = UpSampleBN_feature(
            skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1
        )


    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )

        x_d0 = self.conv2(x_block4)
        x_1_16 = self.up16(x_d0, x_block3)
        x_1_8 = self.up8(x_1_16, x_block2)
        x_1_4 = self.up4(x_1_8, x_block1)
        x_1_2 = self.up2(x_1_4, x_block0)
        x_1_1 = self.up1(x_1_2, features[0])

        x_1_16_feature = self.up16_feature(x_d0, x_block3)
        x_1_8_feature = self.up8_feature(x_1_16_feature, x_block2)
        x_1_4_feature = self.up4_feature(x_1_8_feature, x_block1)
        x_1_2_feature = self.up2_feature(x_1_4_feature, x_block0)
        x_1_1_feature = self.up1_feature(x_1_2_feature, features[0])

        return {
            "1_1": x_1_1_feature + x_1_1,
            "1_2": x_1_2_feature + x_1_2,
            "1_4": x_1_4_feature + x_1_4,
            "1_8": x_1_8_feature + x_1_8,
            "1_16": x_1_16_feature + x_1_16,
            "1_32": x_d0,
            "2_1": x_1_1
        }

def _PSP1x1Conv2d(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


class _PyramidPooling2d(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling2d, self).__init__()
        out_channels = int(in_channels / 2)
        self.avgpool1 = nn.AdaptiveAvgPool2d((240, 320))
        self.avgpool2 = nn.AdaptiveAvgPool2d((120, 160))
        self.avgpool3 = nn.AdaptiveAvgPool2d((60, 80))
        self.avgpool4 = nn.AdaptiveAvgPool2d((30, 40))
        self.conv1 = _PSP1x1Conv2d(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv2d(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv2d(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)

        return torch.cat([x, feat1, feat2], dim=1)
class UNet2D_NYU(nn.Module):
    def __init__(self, backend, num_features, out_feature, use_decoder=True):
        super(UNet2D_NYU, self).__init__()
        self.use_decoder = use_decoder
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN_NYU(
            out_feature=out_feature,
            use_decoder=use_decoder,
            bottleneck_features=num_features,
            num_features=num_features,
        )
        self.psp = _PyramidPooling2d(128, norm_layer=nn.BatchNorm2d, norm_kwargs=None)

        self.seg_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.seg_bn2 = nn.BatchNorm2d(128)
        self.seg_relu2 = nn.ReLU()
        self.seg_drop = nn.Dropout(0.3)
        self.seg_conv2 = nn.Conv2d(128, 12, kernel_size=3, padding=1)

        '''self.reg_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.reg_bn2 = nn.BatchNorm2d(128)
        self.reg_relu2 = nn.ReLU()'''
        self.reg_conv2 = nn.Conv2d(128, 60, kernel_size=3, padding=1)

    def forward(self, x, **kwargs):
        encoded_feats = self.encoder(x)
        unet_out = self.decoder(encoded_feats, **kwargs)

        seg_out = self.psp(unet_out['1_1'])
        seg_out1 = self.seg_relu2(self.seg_bn2(self.seg_conv(seg_out)))
        seg_out1 = self.seg_drop(seg_out1)
        unet_out['1_1'] = seg_out1
        seg_out = self.seg_conv2(seg_out1)

        reg_out = self.reg_conv2(seg_out1)
        reg_out = F.softmax(reg_out, dim=1)
        reg_out = depthregression_NYU()(reg_out)

        return unet_out, seg_out, reg_out*0.1


    def get_encoder_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_decoder_params(self):  # lr learning rate
        return self.decoder.parameters()

    @classmethod
    def build(cls, **kwargs):
        basemodel_name = "tf_efficientnet_b7_ns"
        num_features = 2560

        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
        )
        print("Done.")

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, num_features=num_features, **kwargs)
        print("Done.")
        return m


class depthregression(nn.Module):
    def __init__(self, maxdepth=64):
        super(depthregression, self).__init__()
        # self.disp = Variable(torch.Tensor(np.reshape(np.array(range(1, 1+maxdepth)), [1, maxdepth, 1, 1])).cuda(),
        #                      requires_grad=False)
        self.disp = torch.arange(1, 1+maxdepth, device='cuda', requires_grad=False).float()[None, :, None, None]

    def forward(self, x):
        # disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * self.disp, 1)
        return out
class depthregression_NYU(nn.Module):
    def __init__(self, maxdepth=60):
        super(depthregression_NYU, self).__init__()
        # self.disp = Variable(torch.Tensor(np.reshape(np.array(range(1, 1+maxdepth)), [1, maxdepth, 1, 1])).cuda(),
        #                      requires_grad=False)
        self.disp = torch.arange(1, 1+maxdepth, device='cuda', requires_grad=False).float()[None, :, None, None]

    def forward(self, x):
        # disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * self.disp, 1)
        return out