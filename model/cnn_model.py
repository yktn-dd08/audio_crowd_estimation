import math

import torch
import torch.nn as nn
import numpy as np
from torchaudio.transforms import MelSpectrogram
from common.logger import get_logger

_STFT_WINDOW_LENGTH_SECONDS = 0.025
_STFT_HOP_LENGTH_SECONDS = 0.010
_LOG_OFFSET = 1.0e-20
_MEL_MIN_HZ = 125
_MEL_MAX_HZ = 7500
_NUM_BANDS = 64
logger = get_logger('model.cnn_model')

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


class SimpleCNN2(nn.Module):
    def __init__(
            self,
            freq_num: int,
            frame_num: int,
            kernel_size: int | list[int],
            dilation_size: int | list[int],
            layer_num: int,
            inter_ch: int | list[int],
            pool_size: int | list[int] = 2,
            out_features=1
    ):
        super(SimpleCNN2, self).__init__()

        kernel_size = SimpleCNN2.set_list('kernel_size', kernel_size, layer_num)
        dilation_size = SimpleCNN2.set_list('dilation_size', dilation_size, layer_num)
        inter_ch = SimpleCNN2.set_list('inter_ch', inter_ch, layer_num)
        pool_size = SimpleCNN2.set_list('pool_size', pool_size, layer_num)
        # if not isinstance(kernel_size, list):
        #     kernel_size = [kernel_size] * layer_num
        # else:
        #     assert len(kernel_size) == layer_num, f'Input List[int] with {layer_num} as kernel_size.'
        # if not isinstance(dilation_size, list):
        #     dilation_size = [dilation_size] * layer_num
        # else:
        #     assert len(dilation_size) == layer_num, f'Input List[int] with {layer_num} as dilation_size.'
        # if not isinstance(inter_ch, list):
        #     inter_ch = [inter_ch] * layer_num
        # else:
        #     assert len(inter_ch) == layer_num, f'Input List[int] with {layer_num} as inter_ch.'
        # if not isinstance(pool_size, list):
        #     pool_size = [pool_size] * layer_num
        # else:
        #     assert len(pool_size) == layer_num, f'Input List[int] with {layer_num} as pool_size.'

        self.conv = nn.ModuleList()
        for l in range(layer_num):
            self.conv.append(
                nn.Conv1d(
                    in_channels=freq_num if l == 0 else inter_ch[l-1],
                    out_channels=inter_ch[l],
                    kernel_size=kernel_size[l],
                    dilation=dilation_size[l]
                )
            )
            self.conv.append(nn.ReLU())
            self.conv.append(nn.MaxPool1d(kernel_size=pool_size[l]))
        reg_time = frame_num
        for l in range(layer_num):
            reg_time = reg_time - dilation_size[l] * (kernel_size[l] - 1)
            reg_time = int(reg_time / pool_size[l])

        self.regressor = nn.Linear(in_features=reg_time * inter_ch[-1], out_features=out_features)

    def forward(self, x):
        for mod in self.conv:
            x = mod(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x

    @staticmethod
    def set_list(param_name, param, list_size):
        if not isinstance(param, list):
            param = [param] * list_size
        else:
            assert len(param) == list_size, f'Input List[int] with {list_size} elements for {param_name}.'
        return param

    @staticmethod
    def is_valid(
            freq_num: int,
            frame_num: int,
            kernel_size: int | list[int],
            dilation_size: int | list[int],
            layer_num: int,
            inter_ch: int | list[int],
            pool_size: int | list[int] = 2
    ):
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * layer_num
        elif len(kernel_size) != layer_num:
            logger.warning(f'Input List[int] with {layer_num} as kernel_size.')
            return False
        if not isinstance(dilation_size, list):
            dilation_size = [dilation_size] * layer_num
        elif len(dilation_size) != layer_num:
            logger.warning(f'Input List[int] with {layer_num} as dilation_size.')
            return False
        if not isinstance(inter_ch, list):
            inter_ch = [inter_ch] * layer_num
        elif len(inter_ch) != layer_num:
            logger.warning(f'Input List[int] with {layer_num} as inter_ch.')
            return False
        if not isinstance(pool_size, list):
            pool_size = [pool_size] * layer_num
        elif len(pool_size) != layer_num:
            logger.warning(f'Input List[int] with {layer_num} as pool_size.')
            return False

        model_params = {
            'freq_num': freq_num,
            'frame_num': frame_num,
            'kernel_size': kernel_size,
            'dilation_size': dilation_size,
            'layer_num': layer_num,
            'inter_ch': inter_ch,
            'pool_size': pool_size
        }
        reg_time = frame_num
        ok_layer_num = 0
        for l in range(layer_num):
            reg_time = reg_time - dilation_size[l] * (kernel_size[l] - 1)
            reg_time = int(reg_time / pool_size[l])
            if reg_time > 0:
                ok_layer_num = l
        if reg_time < 1:
            logger.warning(f'Invalid model parameters, {model_params}. You should input {ok_layer_num + 1} as `layer_num` maybe.')
            return False
        return True


class MultiTaskCNN(nn.Module):
    def __init__(
            self,
            task_num: int,
            freq_num:int,
            frame_num:int,
            kernel_size: int | list[int] | list[list[int]],
            dilation_size: int | list[int] | list[list[int]],
            layer_num: int | list[int],
            inter_ch: int | list[int] | list[list[int]],
            pool_size: int | list[int] | list[list[int]] = 2,
            out_features=1
    ):
        super(MultiTaskCNN, self).__init__()
        layer_num = MultiTaskCNN.set_list('layer_num', layer_num, task_num)

    @staticmethod
    def set_list(param_name, param, list_size: int):
        if not isinstance(param, list):
            param = [param] * list_size
        else:
            assert len(param) == list_size, f'Input List[int] with {list_size} elements for {param_name}.'
        return param

    @staticmethod
    def set_double_list(param_name, param, task_num: int, layer_num: list[int]):
        assert task_num == len(layer_num), f'Input list[int] with {task_num} elements for layer_num.'
        if not isinstance(param, list):
            param = [[param] * layer_num[i] for i in range(task_num)]
        else:
            assert len(param) == task_num, f'Input list with {task_num} elements for {param_name}.'
            tmp = []
            for i in range(task_num):
                pass
            if not isinstance(param[0], list):
                param = [[param[i]] * layer_num[i] for i in range(task_num)]
            else:
                for i in range(task_num):
                    pass
            pass
        return param