import torch
import torch.nn as nn


class LeastSquareModel(nn.Module):
    def __init__(self, power=False, log=False, out_features=1):
        super(LeastSquareModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=out_features)
        self.power = power
        self.log = log

    def forward(self, x):
        if self.power:
            x = torch.exp(x)
        x = x.mean(axis=[1, 2]).unsqueeze(1)
        if self.power:
            x = torch.log(x)
        x = self.linear(x)
        if self.log:
            x = torch.log(x)
        return x
