import torch
import torch.nn as nn

from .modules import BidirectionalLSTM, ResNet_FeatureExtractor


class RecognitionModel(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, hidden_size: int, num_class: int):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    # def forward(self, input, text): NOTE: text var not used
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
