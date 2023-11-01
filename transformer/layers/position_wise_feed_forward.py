import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ffn, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layer(x)
