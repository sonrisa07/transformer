import torch
import torch.nn as nn
from transformer.embedding import TransformerEmbedding
from transformer.models.decoder import Decoder
from transformer.models.encoder import Encoder
from data_process import enc_pad_idx, dec_pad_idx


class Transformer(nn.Module):

    def __init__(self, d_model, d_k, d_v, d_ffn, n_heads, n_encoder_layers, n_decoder_layers,
                 enc_vocab_size, dec_vocab_size, enc_max_len, dec_max_len, dropout, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, d_k, d_v, d_ffn, n_heads, n_encoder_layers, dropout)
        self.decoder = Decoder(dec_vocab_size, d_model, d_k, d_v, d_ffn, n_heads, n_decoder_layers, dropout)
        self.enc_embedding = TransformerEmbedding(enc_vocab_size, d_model, enc_max_len, dropout, enc_pad_idx, device)
        self.dec_embedding = TransformerEmbedding(dec_vocab_size, d_model, dec_max_len, dropout, dec_pad_idx, device)
        self.device = device

    def forward(self, enc_input, dec_input):
        enc_mask = (enc_input != enc_pad_idx).unsqueeze(1).to(self.device)
        dec_mask = (dec_input != dec_pad_idx).unsqueeze(1).to(self.device)
        auto_regressive_mask = (torch.tril(torch.ones(dec_input.shape[1], dec_input.shape[1], dtype=torch.bool))
                                .to(self.device))
        x = self.encoder(self.enc_embedding(enc_input), enc_mask)
        x = self.decoder(self.dec_embedding(dec_input), x, enc_mask, dec_mask & auto_regressive_mask)
        return x
