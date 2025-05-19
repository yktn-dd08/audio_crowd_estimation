import torch
import torch.nn as nn

from torchaudio.prototype.pipelines import VGGISH
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
from torch.nn import Linear


class VGGishLinear(nn.Module):
    """
    VGGish + 線型結合での回帰用モデル
    現状、1バッチに対して1秒のデータしか入力できない
    x: Tensor[batch_size x 1 x 96 x 64]
    """
    def __init__(self, vgg_frame=1, out_features=1, pre_trained=True):
        super(VGGishLinear, self).__init__()
        self.input_proc = VGGISH.get_input_processor()
        self.vggish = VGGISH.get_model() if pre_trained else VGGISH.VGGish()
        self.time_sec = vgg_frame
        self.linear = Linear(in_features=vgg_frame * 128, out_features=out_features)
        if pre_trained:
            for param in self.vggish.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x = self.input_proc(x)
        x = self.vggish(x)
        # assert x.size()[0] == self.vgg_frame
        x = self.linear(torch.flatten(x, 1))
        return x


class VGGishLinear2(nn.Module):
    """
    VGGish + 線型結合での回帰用モデル

    VGGishLinearで対応できなかった複数フレームに対応、ただしこの修正方法が良いかは不明

    x: Tensor [batch_size x frame_num x 1 x 96 x 64] -> batched input
       Tensor [frame_num x 1 x 96 x 64]              -> unbatched input
    """
    def __init__(self, frame_num=5, out_features=1, pre_trained=True):
        super(VGGishLinear2, self).__init__()
        self.input_proc = VGGISH.get_input_processor()
        self.vggish = VGGISH.get_model() if pre_trained else VGGISH.VGGish()
        self.time_sec = frame_num
        self.regressor = nn.Sequential(
            Linear(in_features=frame_num * 128, out_features=frame_num * 128),
            nn.ReLU(),
            Linear(in_features=frame_num * 128, out_features=out_features)
        )
        if pre_trained:
            for param in self.vggish.parameters():
                param.requires_grad = False

    def forward(self, x):
        if len(x.size()) == 5:
            # batched input
            x = torch.stack([self.vggish(xx) for xx in x])
            x = torch.flatten(x, 1)
            x = self.regressor(x)
        else:
            # unbatched input
            x = self.vggish(x)
            x = torch.flatten(x)
            x = self.regressor(x)
        return x


class PositionalEncoding(nn.Module):
    """
    References: https://github.com/urbanaudiosensing/Models/blob/main/models/model.py
    """
    def __init__(self, segment_length, token_dim):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Embedding(segment_length, token_dim)
        self.pe_input = torch.Tensor(range(segment_length)).long()

    def forward(self, x):
        x = x + self.pe(self.pe_input.to(x.device))
        return x


class VGGishTransformer(nn.Module):
    """
    VGGish + Transformerでの回帰用モデル

    x: Tensor [batch_size x frame_num x 1 x 96 x 64] -> batched input
       Tensor [frame_num x 1 x 96 x 64]              -> unbatched input
    """
    def __init__(self, frame_num=10, token_dim=128, n_head=4, h_dim=128, layer_num=1, out_features=1, pre_trained=True):
        super(VGGishTransformer, self).__init__()
        self.input_proc = VGGISH.get_input_processor()
        self.vggish = VGGISH.get_model() if pre_trained else VGGISH.VGGish()
        self.pos_encoding = PositionalEncoding(segment_length=frame_num, token_dim=token_dim)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=token_dim, nhead=n_head, dim_feedforward=h_dim, batch_first=True),
            num_layers=layer_num
        )
        self.regressor = nn.Sequential(
            Linear(in_features=h_dim, out_features=h_dim),
            nn.ReLU(),
            Linear(in_features=h_dim, out_features=out_features)
        )

    def forward(self, x):
        if len(x.size()) == 5:
            # batched input
            x = torch.stack([self.vggish(xx) for xx in x])  # batch_size x frame_num x 128
            x = self.pos_encoding(x)
            x = self.transformer(x)[:, -1]
            x = self.regressor(x)
        else:
            # unbatched input
            x = self.vggish(x)  # frame_num x 128
            x = self.pos_encoding(x)
            x = self.transformer(x)[-1]
            x = self.regressor(x)

        return x
