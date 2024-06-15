import torch
import torch.nn as nn
import torch.optim as optim
import math


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=32,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.decoder_embedding = nn.Linear(output_dim, d_model)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.init_weights()

    def init_weights(self):
        # 使用Xavier初始化适用于所有线性层
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.xavier_uniform_(self.decoder_embedding.weight)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.decoder_embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output
