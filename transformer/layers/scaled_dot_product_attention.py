import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v, mask):
        score = (q @ torch.transpose(k, -1, -2)) / (q.shape[-2] ** 0.5)
        if mask is not None:
            score = score.masked_fill(mask == 0, torch.finfo(torch.float32).min)
        return self.softmax(score) @ v
