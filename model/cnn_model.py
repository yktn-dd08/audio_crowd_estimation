import math
import torch
import torch.nn as nn
import numpy as np
from torchaudio.transforms import MelSpectrogram
_STFT_WINDOW_LENGTH_SECONDS = 0.025
_STFT_HOP_LENGTH_SECONDS = 0.010
_LOG_OFFSET = 1.0e-20
_MEL_MIN_HZ = 125
_MEL_MAX_HZ = 7500
_NUM_BANDS = 64

"""
_EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
_EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.
"""

def trans_logmel(signal: np.ndarray | torch.Tensor, fs=16000):
    if isinstance(signal, np.ndarray):
        signal = torch.Tensor(signal)
    window_length_samples = int(round(fs * _STFT_WINDOW_LENGTH_SECONDS))
    hop_length_samples = int(round(fs * _STFT_HOP_LENGTH_SECONDS))
    fft_length = 2 ** int(math.ceil(math.log(window_length_samples) / math.log(2.0)))
    mel_spec = MelSpectrogram(
        sample_rate=fs,
        n_fft=fft_length,
        hop_length=hop_length_samples,
        f_min=_MEL_MIN_HZ,
        f_max=_MEL_MAX_HZ,
        n_mels=_NUM_BANDS
    )
    return torch.log(mel_spec(signal) + _LOG_OFFSET)


class SimpleCNN(nn.Module):
    """
    Simple 1D-CNN model
    x: Tensor [batch_size x freq_num x frame_num]
    """
    def __init__(self, freq_num, frame_num, internal_ch=None, kernel_size=21):
        super(SimpleCNN, self).__init__()
        if internal_ch is None:
            internal_ch = [32, 32]
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=freq_num, out_channels=internal_ch[0], kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=internal_ch[0], out_channels=internal_ch[1], kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        reg_time = int((int((frame_num - kernel_size + 1) / 2) - kernel_size + 1) / 2)
        self.regressor = nn.Linear(in_features=reg_time * internal_ch[1], out_features=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x
