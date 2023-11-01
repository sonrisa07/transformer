import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model, device):
        super(PositionalEncoding, self).__init__()
        self.mapping = torch.zeros([max_len, d_model], requires_grad=False).to(device)
        pos = torch.arange(0, max_len, device=device, dtype=torch.float32, requires_grad=False).unsqueeze(dim=1)
        col = torch.tensor([1000 ** (2 * i / d_model) for i in range(0, d_model, 2)], device=device,
                           dtype=torch.float32, requires_grad=False)
        self.mapping[:, 0::2] = torch.sin(pos / col)
        self.mapping[:, 1::2] = torch.cos(pos / col)

    def forward(self, x):
        return self.mapping[0:x.shape[-1]]
