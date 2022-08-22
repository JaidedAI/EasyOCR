from collections import OrderedDict
import pdb
import torch
import torch.nn as nn
from .feature_attention import ScaleFeatureSelection
BatchNorm2d = nn.BatchNorm2d


class SegSpatialScaleDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,fpn=True, attention_type='scale_spatial',
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegSpatialScaleDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.fpn = fpn
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        if self.fpn:
            self.out5 = nn.Sequential(
                nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
                nn.Upsample(scale_factor=8, mode='nearest'))
            self.out4 = nn.Sequential(
                nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
                nn.Upsample(scale_factor=4, mode='nearest'))
            self.out3 = nn.Sequential(
                nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
                nn.Upsample(scale_factor=2, mode='nearest'))
            self.out2 = nn.Conv2d(inner_channels, inner_channels//4, 3, padding=1, bias=bias)
            self.out5.apply(self.weights_init)
            self.out4.apply(self.weights_init)
            self.out3.apply(self.weights_init)
            self.out2.apply(self.weights_init)

            self.concat_attention = ScaleFeatureSelection(inner_channels, inner_channels//4, attention_type=attention_type)
            self.binarize = nn.Sequential(
                nn.Conv2d(inner_channels, inner_channels // 4, 3, bias=bias, padding=1),
                BatchNorm2d(inner_channels//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
                BatchNorm2d(inner_channels//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
                nn.Sigmoid())
        else:
            self.concat_attention = ScaleFeatureSelection(inner_channels, inner_channels//4, )
            self.binarize = nn.Sequential(
                nn.Conv2d(inner_channels, inner_channels // 4, 3, bias=bias, padding=1),
                BatchNorm2d(inner_channels//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
                BatchNorm2d(inner_channels//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
                nn.Sigmoid())

        self.binarize.apply(self.weights_init)
        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.concat_attention(fuse, [p5, p4, p3, p2])
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))