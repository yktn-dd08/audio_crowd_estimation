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


def set_list(param_name, param, list_size):
    if not isinstance(param, list):
        param = [param] * list_size
    else:
        assert len(param) == list_size, f'Input List[int] with {list_size} elements for {param_name}.'
    return param


def set_double_list(param_name, param, task_num: int, layer_num: list[int]):
    assert task_num == len(layer_num), f'Input list[int] with {task_num} elements for layer_num.'
    if not isinstance(param, list):
        param = [[param] * layer_num[i] for i in range(task_num)]
    else:
        assert len(param) == task_num, f'Input list with {task_num} elements for {param_name}.'
        tmp = []
        for i in range(task_num):
            tmp.append(param[i] if isinstance(param[i], list) else [param[i]] * layer_num[i])
        param = tmp
    return param


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

        kernel_size = set_list('kernel_size', kernel_size, layer_num)
        dilation_size = set_list('dilation_size', dilation_size, layer_num)
        inter_ch = set_list('inter_ch', inter_ch, layer_num)
        pool_size = set_list('pool_size', pool_size, layer_num)
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

    # @staticmethod
    # def set_list(param_name, param, list_size):
    #     if not isinstance(param, list):
    #         param = [param] * list_size
    #     else:
    #         assert len(param) == list_size, f'Input List[int] with {list_size} elements for {param_name}.'
    #     return param

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

"""
from model.cnn_model import *
batch_size = 32
freq_num = 64
frame_num = 500

layer_num = 3
task_num = 5
kernel_size = 3
dilation_size = 3
inter_ch = 4
pool_size = 1
model = MultiTaskCNN(task_num=task_num, freq_num=freq_num, frame_num=frame_num, kernel_size=kernel_size,
                     dilation_size=dilation_size, layer_num=layer_num, inter_ch=inter_ch, pool_size=pool_size)
x_np = np.random.random([batch_size, freq_num, frame_num])
x = torch.Tensor(x_np)
task = model.each_task(x, 0)
"""

class AreaSpecificCNN(nn.Module):
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
        super(AreaSpecificCNN, self).__init__()
        self.task_num = task_num
        layer_num = set_list('layer_num', layer_num, task_num)
        kernel_size = set_double_list('kernel_size', kernel_size, task_num, layer_num)
        dilation_size = set_double_list('dilation_size', dilation_size, task_num, layer_num)
        inter_ch = set_double_list('inter_ch', inter_ch, task_num, layer_num)
        pool_size = set_double_list('pool_size', pool_size, task_num, layer_num)

        self.task_conv = nn.ModuleList()
        self.task_regressor = nn.ModuleList()

        for t in range(task_num):
            each_conv = nn.ModuleList()
            for l in range(layer_num[t]):
                each_conv.append(nn.Conv1d(in_channels=freq_num if l == 0 else inter_ch[t][l-1],
                                           kernel_size=kernel_size[t][l],
                                           dilation=dilation_size[t][l],
                                           out_channels=inter_ch[t][l]))
                each_conv.append(nn.ReLU())
                each_conv.append(nn.MaxPool1d(kernel_size=pool_size[t][l]))
            self.task_conv.append(each_conv)

            reg_time = frame_num
            for l in range(layer_num[t]):
                reg_time = reg_time - dilation_size[t][l] * (kernel_size[t][l] - 1)
                reg_time = int(reg_time / pool_size[t][l])
            # TODO
            #  AsIs: 全特徴量をフラット化して推定
            #  ToBe: 1D-CNNにして時間報告を集約したのち、チャネル方向の集約を行なって各タスクを推定
            self.task_regressor.append(nn.Linear(in_features=reg_time * inter_ch[t][-1], out_features=1))
        # TODO
        #  AsIs: 各タスクの出力の線型結合により人数を推定
        #  ToBe: 各タスクの推定値を算出する直前の出力(チャネル方向の情報は保持したもの)を結合して線型結合により人数を推定
        self.regressor = nn.Linear(in_features=task_num, out_features=out_features)

    # @staticmethod
    # def set_list(param_name, param, list_size: int):
    #     if not isinstance(param, list):
    #         param = [param] * list_size
    #     else:
    #         assert len(param) == list_size, f'Input List[int] with {list_size} elements for {param_name}.'
    #     return param
    #
    # @staticmethod
    # def set_double_list(param_name, param, task_num: int, layer_num: list[int]):
    #     assert task_num == len(layer_num), f'Input list[int] with {task_num} elements for layer_num.'
    #     if not isinstance(param, list):
    #         param = [[param] * layer_num[i] for i in range(task_num)]
    #     else:
    #         assert len(param) == task_num, f'Input list with {task_num} elements for {param_name}.'
    #         tmp = []
    #         for i in range(task_num):
    #             tmp.append(param[i] if isinstance(param[i], list) else [param[i]] * layer_num[i])
    #         param = tmp
    #     return param
    
    @staticmethod
    def is_valid(task_num: int,
                 freq_num:int,
                 frame_num:int,
                 kernel_size: int | list[int] | list[list[int]],
                 dilation_size: int | list[int] | list[list[int]],
                 layer_num: int | list[int],
                 inter_ch: int | list[int] | list[list[int]],
                 pool_size: int | list[int] | list[list[int]] = 2,
                 out_features=1):
        try:
            layer_num = set_list('layer_num', layer_num, task_num)
            kernel_size = set_double_list('kernel_size', kernel_size, task_num, layer_num)
            dilation_size = set_double_list('dilation_size', dilation_size, task_num, layer_num)
            inter_ch = set_double_list('inter_ch', inter_ch, task_num, layer_num)
            pool_size = set_double_list('pool_size', pool_size, task_num, layer_num)
        except AssertionError as e:
            logger.info(f'Not valid model: {e}.')
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

        for t in range(task_num):
            ok_layer_num = 0
            reg_time = frame_num
            for l in range(layer_num[t]):
                reg_time = reg_time - dilation_size[t][l] * (kernel_size[t][l] - 1)
                reg_time = int(reg_time / pool_size[t][l])
                if reg_time > 0:
                    ok_layer_num = l
            if reg_time < 1:
                logger.warning(f'Invalid model parameters, {model_params}. You should input {ok_layer_num + 1} as `layer_num[{t}]` maybe.')
                return False
        return True

    def each_task(self, x, task_idx):
        for mod in self.task_conv[task_idx]:
            x = mod(x)
        # TODO unbatch_modeの場合の実装（そもそもいるのか？）
        x = torch.flatten(x, 1)
        reg = self.task_regressor[task_idx]
        x = reg(x)
        return x

    def task(self, x):
        x = torch.cat([self.each_task(x, t) for t in range(self.task_num)], dim=1)
        return x

    def forward(self, x):
        shared = self.task(x)
        y = self.regressor(shared)
        return y, shared


class AreaSpecificCNN2(nn.Module):
    def __init__(
            self,
            task_num: int,
            freq_num: int,
            frame_num: int,

            common_kernel_size: int | list[int],
            kernel_size: int | list[int] | list[list[int]],

            common_dilation_size: int | list[int],
            dilation_size: int | list[int] | list[list[int]],

            common_layer_num: int,
            layer_num: int | list[int],

            common_inter_ch: int | list[int],
            inter_ch: int | list[int] | list[list[int]],

            common_pool_size: int | list[int] = 2,
            pool_size: int | list[int] | list[list[int]] = 2,

            out_features=1
    ):
        super(AreaSpecificCNN2, self).__init__()

        common_kernel_size = set_list('kernel_size', common_kernel_size, common_layer_num)
        common_dilation_size = set_list('dilation_size', common_dilation_size, common_layer_num)
        common_inter_ch = set_list('inter_ch', common_inter_ch, common_layer_num)
        common_pool_size = set_list('pool_size', common_pool_size, common_layer_num)

        self.conv = nn.ModuleList()
        for l in range(common_layer_num):
            self.conv.append(
                nn.Conv1d(
                    in_channels=freq_num if l == 0 else common_inter_ch[l-1],
                    out_channels=common_inter_ch[l],
                    kernel_size=common_kernel_size[l],
                    dilation=common_dilation_size[l]
                )
            )
            self.conv.append(nn.ReLU())
            self.conv.append(nn.MaxPool1d(kernel_size=common_pool_size[l]))
        common_reg_time = frame_num
        for l in range(common_layer_num):
            common_reg_time = common_reg_time - common_dilation_size[l] * (common_kernel_size[l] - 1)
            common_reg_time = int(common_reg_time / common_pool_size[l])

        layer_num = set_list('layer_num', layer_num, task_num)
        kernel_size = set_double_list('kernel_size', kernel_size, task_num, layer_num)
        dilation_size = set_double_list('dilation_size', dilation_size, task_num, layer_num)
        inter_ch = set_double_list('inter_ch', inter_ch, task_num, layer_num)
        pool_size = set_double_list('pool_size', pool_size, task_num, layer_num)

        self.task_num = task_num
        self.task_conv = nn.ModuleList()
        self.task_regressor = nn.ModuleList()

        for t in range(task_num):
            each_conv = nn.ModuleList()
            for l in range(layer_num[t]):
                each_conv.append(nn.Conv1d(in_channels=common_inter_ch[-1] if l == 0 else inter_ch[t][l-1],
                                           kernel_size=kernel_size[t][l],
                                           dilation=dilation_size[t][l],
                                           out_channels=inter_ch[t][l]))
                each_conv.append(nn.ReLU())
                each_conv.append(nn.MaxPool1d(kernel_size=pool_size[t][l]))
            self.task_conv.append(each_conv)

            reg_time = common_reg_time
            for l in range(layer_num[t]):
                reg_time = reg_time - dilation_size[t][l] * (kernel_size[t][l] - 1)
                reg_time = int(reg_time / pool_size[t][l])
            # TODO
            #  AsIs: 全特徴量をフラット化して推定
            #  ToBe: 1D-CNNにして時間報告を集約したのち、チャネル方向の集約を行なって各タスクを推定
            self.task_regressor.append(nn.Linear(in_features=reg_time * inter_ch[t][-1], out_features=1))
        # TODO
        #  AsIs: 各タスクの出力の線型結合により人数を推定
        #  ToBe: 各タスクの推定値を算出する直前の出力(チャネル方向の情報は保持したもの)を結合して線型結合により人数を推定
        self.regressor = nn.Linear(in_features=task_num, out_features=out_features)

    @staticmethod
    def is_valid(
        task_num: int,
        freq_num: int,
        frame_num: int,

        common_kernel_size: int | list[int],
        kernel_size: int | list[int] | list[list[int]],

        common_dilation_size: int | list[int],
        dilation_size: int | list[int] | list[list[int]],

        common_layer_num: int,
        layer_num: int | list[int],

        common_inter_ch: int | list[int],
        inter_ch: int | list[int] | list[list[int]],

        common_pool_size: int | list[int] = 2,
        pool_size: int | list[int] | list[list[int]] = 2,

        out_features=1
    ):
        try:
            common_kernel_size = set_list('kernel_size', common_kernel_size, common_layer_num)
            common_dilation_size = set_list('dilation_size', common_dilation_size, common_layer_num)
            common_inter_ch = set_list('inter_ch', common_inter_ch, common_layer_num)
            common_pool_size = set_list('pool_size', common_pool_size, common_layer_num)

            layer_num = set_list('layer_num', layer_num, task_num)
            kernel_size = set_double_list('kernel_size', kernel_size, task_num, layer_num)
            dilation_size = set_double_list('dilation_size', dilation_size, task_num, layer_num)
            inter_ch = set_double_list('inter_ch', inter_ch, task_num, layer_num)
            pool_size = set_double_list('pool_size', pool_size, task_num, layer_num)
        except AssertionError as e:
            logger.info(f'Not valid model: {e}.')
            return False

        model_params = {
            'freq_num': freq_num,
            'frame_num': frame_num,
            'common_kernel_size': common_kernel_size,
            'common_dilation_size': common_dilation_size,
            'common_layer_num': common_layer_num,
            'common_inter_ch': common_inter_ch,
            'common_pool_size': common_pool_size
        }
        common_reg_time = frame_num
        for l in range(common_layer_num):
            common_reg_time = common_reg_time - common_dilation_size[l] * (common_kernel_size[l] - 1)
            common_reg_time = int(common_reg_time / common_pool_size[l])
        if common_reg_time < 1:
            logger.warning(
                f'Invalid model parameters, {model_params}. You should input small number as `common_layer_num`.'
            )
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

        for t in range(task_num):
            reg_time = common_reg_time
            for l in range(layer_num[t]):
                reg_time = reg_time - dilation_size[t][l] * (kernel_size[t][l] - 1)
                reg_time = int(reg_time / pool_size[t][l])
            if reg_time < 1:
                logger.warning(
                    f'Invalid model parameters, {model_params}. You should input small number as `layer_num`'
                )

        return True

    def common_layer(self, x):
        for mod in self.conv:
            x = mod(x)
        return x

    def each_area(self, x, task_idx):
        for mod in self.task_conv[task_idx]:
            x = mod(x)
        x = torch.flatten(x, 1)
        reg = self.task_regressor[task_idx]
        x = reg(x)
        return x

    def forward(self, x):
        x = self.common_layer(x)
        x = torch.cat([self.each_area(x, t) for t in range(self.task_num)], dim=1)
        y = self.regressor(x)
        return y, x
