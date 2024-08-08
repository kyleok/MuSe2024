import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from config import ACTIVATION_FUNCTIONS, device
from .layers import OutLayer, last_item_from_packed


class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)

        if self.n_to_1:
            return last_item_from_packed(rnn_enc[0], x_len)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)
        self.encoder = RNN(params.model_dim, params.model_dim, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)
        # d_in : model_dim, d_out:model_dim, n_layers=1 : rnn_n_layers, bi=True: rnn_bi, dropout=0.2, n_to_1=False

        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        # d_in, d_hidden, d_out, dropout=.0, bias=.0
        # d_in: d_rnn_out, d_hidden: d_fc_out, d_out: n_targets, dropout: linear_dropout
        # self.output_layer = nn.Linear(params.hidden_size, params.vocab_size): xLSTM
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.encoder(x, x_len)
        y = self.out(x)
        activation = self.final_activation(y)
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1
