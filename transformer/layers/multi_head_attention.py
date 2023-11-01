import torch
import torch.nn as nn

from transformer.layers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads)
        self.w_k = nn.Linear(d_model, d_k * n_heads)
        self.w_v = nn.Linear(d_model, d_v * n_heads)
        self.w_o = nn.Linear(d_v * n_heads, d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # in fact, q, k, v are x, x, x from encoder input or x, x, y from cross attention

        # calculate real Q, K, V and split them of each head into tuples, respectively
        Q = torch.chunk(self.w_q(q), self.d_k, -1)
        K = torch.chunk(self.w_k(k), self.d_k, -1)
        V = torch.chunk(self.w_v(v), self.d_v, -1)

        out = []
        for Q_i, K_i, V_i in zip(Q, K, V):
            out.append(self.attention(Q_i, K_i, V_i, mask))
        return self.dropout(self.w_o(torch.concat(out, -1)))

