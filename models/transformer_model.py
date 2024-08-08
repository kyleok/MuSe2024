import torch.nn as nn
import torch
from config import ACTIVATION_FUNCTIONS, device
from .layers import OutLayer, last_item_from_packed
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TFModel(nn.Module):
    def __init__(self, params):
        super(TFModel, self).__init__()
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
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
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

        output = self.out(output)
        output = self.final_activation(output)

        return output,src