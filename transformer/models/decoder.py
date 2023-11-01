import torch.nn as nn

from transformer.modules.decoder_layer import DecoderLayer


class Decoder(nn.Module):

    def __init__(self, dec_vocab_size, d_model, d_k, d_v, d_ffn, n_heads, n_layers, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_k, d_v, d_ffn, n_heads, dropout) for _ in range(n_layers)
        ])
        self.linear = nn.Linear(d_model, dec_vocab_size)

    def forward(self, dec_input, enc_output, enc_mask, dec_mask):
        x = None
        for layer in self.layers:
            x = layer(dec_input, enc_output, enc_mask, dec_mask)
        return self.linear(x)
