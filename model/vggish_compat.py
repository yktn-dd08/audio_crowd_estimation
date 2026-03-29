import math
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from torchvggish import vggish, vggish_input

# torchaudio.prototype の VGGishInputProcessor 相当を簡易再現
class VGGishInputProcessor:
    """
    waveform (T,) を受けて、VGGish に入れる shape に変換する。
    返り値の shape は (n_example, 1, n_frame, 64) を想定。
    torchaudio 2.8 docs の VGGishInputProcessor と互換の役割。
    """
    def __call__(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        if waveform.dim() != 1:
            waveform = waveform.flatten()

        # torchvggish の前処理を利用
        # numpy 入力を想定しているので CPU numpy に変換
        examples = vggish_input.waveform_to_examples(
            waveform.detach().cpu().numpy(),
            sample_rate
        )
        # torchvggish は (N, 96, 64) を返すので channel 次元追加
        x = torch.tensor(examples, dtype=torch.float32)
        x = x.unsqueeze(1)  # (N, 1, 96, 64)
        return x


class VGGishModel(nn.Module):
    """
    torchaudio.prototype.pipelines.VGGISH.get_model() 的に使うための wrapper
    """
    def __init__(self):
        super().__init__()
        self.model = vggish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchvggish 側は通常 (N, 1, 96, 64) を受ける
        return self.model(x)


class _VGGISHBundle:
    """
    torchaudio.prototype.pipelines.VGGISH に似せた bundle
    """
    @staticmethod
    def get_input_processor():
        return VGGishInputProcessor()

    @staticmethod
    def get_model():
        model = VGGishModel()
        model.eval()
        return model


VGGISH = _VGGISHBundle()
