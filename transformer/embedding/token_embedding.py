import torch.nn as nn


class TokenEmbedding(nn.Embedding):

    def __int__(self, vocab_size, _d_model):
        super(TokenEmbedding, self).__init__(vocab_size, _d_model)


