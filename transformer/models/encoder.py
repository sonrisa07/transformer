import torch.nn as nn

from transformer.modules.encoder_layer import EncoderLayer


class Encoder(nn.Module):

    def __init__(self, d_model, d_k, d_v, d_ffn, n_heads, n_layers, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_k, d_v, d_ffn, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
