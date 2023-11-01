from torch import nn

from .positional_encoding import PositionalEncoding
from .token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, dropout, pad_idx, device):
        super(TransformerEmbedding, self).__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_idx, device=device)
        self.pos_emb = PositionalEncoding(max_len, d_model, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.token_emb(x) + self.pos_emb(x))
