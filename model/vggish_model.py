import torch
import torch.nn as nn

from torchaudio.prototype.pipelines import VGGISH
from torch.nn import Transformer
from torch.nn import Linear


class VGGishLinear(nn.Module):
    def __init__(self, time_sec=1, out_features=1, pre_trained=True):
        super(VGGishLinear, self).__init__()
        self.input_proc = VGGISH.get_input_processor()
        self.vggish = VGGISH.get_model()
        self.time_sec = time_sec
        self.linear = Linear(in_features=time_sec * 128, out_features=out_features)
        if pre_trained:
            for param in self.vggish.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.input_proc(x)
        x = self.vggish(x)
        assert x.size()[0] == self.time_sec
        x = self.linear(torch.flatten(x, 1))
        return x


class VGGishTransformer(nn.Module):
    def __init__(self, time_sec=1, out_features=1, pre_trained=True):
        super(VGGishTransformer, self).__init__()
        self.input_proc = VGGISH.get_input_processor()
        self.vggish = VGGISH.get_model()
        self.transformer = Transformer()

    def forward(self, x):
        return self, x
