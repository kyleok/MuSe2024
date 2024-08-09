import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from config import ACTIVATION_FUNCTIONS, device
from .layers import OutLayer, last_item_from_packed


class MultiAttModel(nn.Module):
    def __init__(self, params):
        super(MultiAttModel, self).__init__()
        self.params = params
        
        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        d_encoder_out = params.model_dim
        
        if params.encoder == 'RNN':
            self.encoder = RNN(params.model_dim, params.model_dim, n_layers=params.encoder_n_layers, bi=params.rnn_bi,
                           dropout=params.encoder_dropout, n_to_1=params.n_to_1, device=params.device)
            d_encoder_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        elif params.encoder == 'TF':
            self.encoder = TransformerModel(params.model_dim, params.model_dim, params.nhead, 
                                        n_layers=params.encoder_n_layers, dim_feedforward=params.dim_feedforward, dropout=params.encoder_dropout, device=params.device)

        self.outs = nn.ModuleDict({     ## multi-label regresesion
                attribute: OutLayer(d_encoder_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout) for attribute in params.label_dims
            })
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.encoder(x, x_len)
        if self.params.encoder == 'TF':  # TODO ADDED
            x = torch.mean(x, dim=1) # TODO ADDED

        y, activation = {}, {}
        for attribute in self.params.label_dims:
            y[attribute] = self.outs[attribute](x)
            activation[attribute] = self.final_activation(y[attribute])
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1