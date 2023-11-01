import torch
import torch.nn as nn

from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_k, d_v, d_ffn, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.sublayer1 = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.sublayer2 = PositionWiseFeedForward(d_model, d_ffn, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.sublayer1(x, x, x, mask)
        x = self.norm(x)
        x = x + self.sublayer2(x)
        return self.norm(x)



