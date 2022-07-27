import torch
import torch.nn as nn
from .modules import VGG_FeatureExtractor, BidirectionalLSTM


class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel

        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = torch.mean(visual_feature.permute(0, 3, 1, 2), dim=3)

        contextual_feature = self.SequenceModeling(visual_feature)

        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
