import torch
import torch.nn as nn
import torch.nn.functional as F


from backbones.upsample_head import SimpleUpsampleHead


class SimpleDetectionDecoder(nn.Module):
    def __init__(self, feature_channel=256):
        nn.Module.__init__(self)

        self.feature_channel = feature_channel
        self.head_layer = self.create_head_layer()

        self.pred_layers = nn.ModuleDict(self.create_pred_layers())

    def create_head_layer(self):
        return SimpleUpsampleHead(
            self.feature_channel,
            [self.feature_channel, self.feature_channel // 2, self.feature_channel // 4]
        )

    def create_pred_layer(self, channels):
        return nn.Sequential(
            nn.Conv2d(self.feature_channel // 4, channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def create_pred_layers(self):
        return {}

    def postprocess_pred(self, pred):
        return pred

    def calculate_losses(self, preds, label):
        raise NotImplementedError()

    def forward(self, input, label, meta, train):
        feature = self.head_layer(input)

        pred = {}
        for name, pred_layer in self.pred_layers.items():
            pred[name] = pred_layer(feature)

        if train:
            losses = self.calculate_losses(pred, label)
            pred = self.postprocess_pred(pred)
            loss = sum(losses.values())
            return loss, pred, losses
        else:
            pred = self.postprocess_pred(pred)
            return pred


class SimpleSegDecoder(SimpleDetectionDecoder):
    def create_pred_layers(self):
        return {
            'heatmap': self.create_pred_layer(1)
        }

    def postprocess_pred(self, pred):
        pred['heatmap'] = F.sigmoid(pred['heatmap'])
        return pred

    def calculate_losses(self, pred, label):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']

        heatmap_pred = pred['heatmap']

        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))

        return {
            'heatmap_loss': heatmap_loss,
        }


class SimpleEASTDecoder(SimpleDetectionDecoder):
    def __init__(self, feature_channels=256, densebox_ratio=1000.0, densebox_rescale_factor=512):
        SimpleDetectionDecoder.__init__(self, feature_channels)

        self.densebox_ratio = densebox_ratio
        self.densebox_rescale_factor = densebox_rescale_factor

    def create_pred_layers(self):
        return {
            'heatmap': self.create_pred_layer(1),
            'densebox': self.create_pred_layer(8),
        }

    def postprocess_pred(self, pred):
        pred['heatmap'] = F.sigmoid(pred['heatmap'])
        pred['densebox'] = pred['densebox'] * self.densebox_rescale_factor
        return pred

    def calculate_losses(self, pred, label):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        densebox = label['densebox'] / self.densebox_rescale_factor
        densebox_weight = label['densebox_weight']

        heatmap_pred = pred['heatmap']
        densebox_pred = pred['densebox']

        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))

        densebox_loss = F.mse_loss(densebox_pred, densebox, reduction='none')
        densebox_loss = (densebox_loss * densebox_weight).mean(dim=(1, 2, 3)) * self.densebox_ratio

        return {
            'heatmap_loss': heatmap_loss,
            'densebox_loss': densebox_loss,
        }


class SimpleTextsnakeDecoder(SimpleDetectionDecoder):
    def __init__(self, feature_channels=256, radius_ratio=10.0):
        SimpleDetectionDecoder.__init__(self, feature_channels)

        self.radius_ratio = radius_ratio

    def create_pred_layers(self):
        return {
            'heatmap': self.create_pred_layer(1),
            'radius': self.create_pred_layer(1),
        }

    def postprocess_pred(self, pred):
        pred['heatmap'] = F.sigmoid(pred['heatmap'])
        pred['radius'] = torch.exp(pred['radius'])
        return pred

    def calculate_losses(self, pred, label):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        radius = torch.log(label['radius'] + 1)
        radius_weight = label['radius_weight']

        heatmap_pred = pred['heatmap']
        radius_pred = pred['radius']

        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))

        radius_loss = F.smooth_l1_loss(radius_pred, radius, reduction='none')
        radius_loss = (radius_loss * radius_weight).mean(dim=(1, 2, 3)) * self.radius_ratio

        return {
            'heatmap_loss': heatmap_loss,
            'radius_loss': radius_loss,
        }


class SimpleMSRDecoder(SimpleDetectionDecoder):
    def __init__(self, feature_channels=256, offset_ratio=1000.0, offset_rescale_factor=512):
        SimpleDetectionDecoder.__init__(self, feature_channels)

        self.offset_ratio = offset_ratio
        self.offset_rescale_factor = offset_rescale_factor

    def create_pred_layers(self):
        return {
            'heatmap': self.create_pred_layer(1),
            'offset': self.create_pred_layer(2),
        }

    def postprocess_pred(self, pred):
        pred['heatmap'] = F.sigmoid(pred['heatmap'])
        pred['offset'] = pred['offset'] * self.offset_rescale_factor
        return pred

    def calculate_losses(self, pred, label):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        offset = label['offset'] / self.offset_rescale_factor
        offset_weight = label['offset_weight']

        heatmap_pred = pred['heatmap']
        offset_pred = pred['offset']

        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))
        offset_loss = F.mse_loss(offset_pred, offset, reduction='none')
        offset_loss = (offset_loss * offset_weight).mean(dim=(1, 2, 3)) * self.offset_ratio

        return {
            'heatmap_loss': heatmap_loss,
            'offset_loss': offset_loss,
        }
