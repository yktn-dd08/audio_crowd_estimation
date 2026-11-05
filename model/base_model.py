import torch
import torch.nn as nn


class LeastSquareModel(nn.Module):
    def __init__(self, power=False, log=False, out_features=1):
        super(LeastSquareModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=out_features)
        self.power = power
        self.log = log

    def extract_features(self, x):
        if self.power:
            x = torch.exp(x)
        x = x.mean(axis=[1, 2]).unsqueeze(1)
        if self.power:
            x = torch.log(x)
        return x

    def forward(self, x):
        x = self.linear(self.extract_features(x))
        if self.log:
            x = torch.log(x)
        return x


class LeastSquareModel2(nn.Module):
    """時間平均したメル帯域別特徴量から、切片付き線形回帰で予測する。

    入力: [batch, 64, frames]、特徴量: [batch, 64]。
    power=Trueではexp→時間平均→log、log=Trueでは線形出力にlogを適用。
    """

    def __init__(self, power=False, log=False, out_features=1):
        super().__init__()
        self.linear = nn.Linear(in_features=64, out_features=out_features)
        self.power = power
        self.log = log

    def extract_features(self, x):
        if x.ndim != 3 or x.shape[1] != self.linear.in_features:
            raise ValueError(
                f'Expected [batch, {self.linear.in_features}, frames], got {tuple(x.shape)}.')
        if self.power:
            x = torch.exp(x)
        x = x.mean(dim=2)
        if self.power:
            x = torch.log(x)
        return x

    def forward(self, x):
        x = self.linear(self.extract_features(x))
        if self.log:
            x = torch.log(x)
        return x
