import torch.nn as nn
import torch

from models.rnn_model import RNN
from models.transformer_model import PositionalEncoding

from config import ACTIVATION_FUNCTIONS, device
from .layers import OutLayer, last_item_from_packed


class SimAttRNNModel(nn.Module):
    def __init__(self, params):
        super(SimAttRNNModel, self).__init__()
        self.params = params
        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)
        self.encoder = RNN(params.model_dim, params.model_dim, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)

        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.outs = nn.ModuleDict({     ## multi-label regresesion
                attribute: OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout) for attribute in params.label_dims
            })
        
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.encoder(x, x_len)
        y, activation = {}, {}
        for attribute in self.params.label_dims:
            y[attribute] = self.outs[attribute](x)
            activation[attribute] = self.final_activation(y[attribute])
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1
    

class SimAttTFModel(nn.Module):
    def __init__(self, params):
        super(SimAttTFModel, self).__init__()

        self.params = params
        self.device = device
        self.n_to_1 = params.n_to_1
        d_model = params.model_dim
        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=params.rnn_n_layers,
                                                        dropout=params.rnn_dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=int(params.rnn_n_layers / 2))
        self.pos_encoder = PositionalEncoding(params.model_dim, params.rnn_dropout)

        self.encoder = nn.Sequential(
            nn.Linear(params.d_in, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.outs = nn.ModuleDict({     ## multi-label regresesion
                attribute: OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout) for attribute in params.label_dims
            })
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_len):
        srcmask = self.generate_square_subsequent_mask(src.shape[1]).to(self.device)
        src = self.encoder(src)
        src = self.pos_encoder(src.transpose(0, 1))  # transpose before pos_encoder
        output = self.transformer_encoder(src, srcmask).transpose(0, 1)  # original
        if self.n_to_1:
            # Get the last relevant item for each sequence in the batch
            last_items = torch.stack([output[i, length - 1, :] for i, length in enumerate(src_len)])
            output = last_items

        output = {}
        for attribute in self.params.label_dims:
            output[attribute] = self.outs[attribute](src)
            output[attribute] = self.final_activation(output[attribute])

        return output, src