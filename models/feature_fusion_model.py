import torch.nn as nn
from config import ACTIVATION_FUNCTIONS, device
from .layers import OutLayer


class FeatureFusionModel(nn.Module):
    def __init__(self, params):
        super(FeatureFusionModel, self).__init__()
        self.params = params
        d_encoder_out = params.d_in
        self.out = OutLayer(d_encoder_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x):
        y = self.out(x)
        activation = self.final_activation(y)
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1