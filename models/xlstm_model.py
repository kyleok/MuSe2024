import torch
import torch.nn as nn
from omegaconf import OmegaConf
from dacite import from_dict, Config as DaciteConfig
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from config import ACTIVATION_FUNCTIONS, device
from .layers import OutLayer, last_item_from_packed



class xLSTMModel(nn.Module):
    def __init__(self, params):
        super(xLSTMModel, self).__init__()
        self.params = params
        self.n_to_1 = params.n_to_1

        # Define input projection layer
        self.inp = nn.Linear(params.d_in, params.embedding_dim, bias=False)

        # Define the xLSTM block stack
        xlstm_cfg = f"""
        mlstm_block:
          mlstm:
            conv1d_kernel_size: {params.kernel_size}
            qkv_proj_blocksize: 4
            num_heads: {params.num_heads}
        slstm_block:
          slstm:
            backend: cuda
            num_heads: {params.num_heads}
            conv1d_kernel_size: {params.kernel_size}
            bias_init: powerlaw_blockdependent
          feedforward:
            proj_factor: {params.proj_factor}
            act_fn: gelu
        context_length: {params.context_length}
        num_blocks: {params.num_blocks}
        embedding_dim: {params.embedding_dim}
        slstm_at: {params.slstm_at}
        """
        cfg = OmegaConf.create(xlstm_cfg)
        cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg),
                        config=DaciteConfig(strict=True))
        self.xlstm_stack = xLSTMBlockStack(cfg)

        # Define the output layer
        d_xlstm_out = params.embedding_dim  # Assuming embedding_dim is the output dimension of xLSTM
        self.out = OutLayer(d_xlstm_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)

        # Define the final activation function
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len=None):
        x = self.inp(x)
        x = self.xlstm_stack(x)
        if self.n_to_1:
            # Get the last relevant item for each sequence in the batch
            last_items = torch.stack([x[i, length - 1, :] for i, length in enumerate(x_len)])
            x = last_items
        y = self.out(x)
        activation = self.final_activation(y)
        return activation, x
