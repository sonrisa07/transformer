import torch.nn as nn

from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.position_wise_feed_forward import PositionWiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_k, d_v, d_ffn, n_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.sublayer1 = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.sublayer2 = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.sublayer3 = PositionWiseFeedForward(d_model, d_ffn, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_input, encoder_output, enc_mask, dec_mask):
        """

        :param dec_input: input of decoder's sublayer
        :param encoder_output: output of encoder's sublayer (named memory)
        :param enc_mask: mask matrix of decoder's input embedding (optional)
        :param dec_mask: mask matrix of decoder's input embedding (necessary for auto-regressive property)
        :return: output tensor of a decoder layer
        """

        x = dec_input + self.sublayer1(dec_input, dec_input, dec_input, dec_mask)
        x = self.norm(x)

        x = x + self.sublayer2(x, encoder_output, encoder_output, enc_mask)
        x = self.norm(x)

        x = x + self.sublayer3(x)
        return self.norm(x)
