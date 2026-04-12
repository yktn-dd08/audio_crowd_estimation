import re
import json
import glob
import argparse
import os.path
import shutil

import optuna
import librosa
import numpy as np
import pandas as pd
import scipy as sp
import soundfile as sf
import torch
import torch.nn as nn
from inspect import signature
from sklearn.model_selection import train_test_split

from model.cnn_model import *
from model.transformer import *
from model.vggish_model import *
from model.base_model import *
# from torchaudio.prototype.pipelines import VGGISH
from model.vggish_compat import VGGISH
from common.logger import get_logger
from analysis.model_common import *


FS = 16000
MODEL_LIST = [
    'VGGishLinear',
    'VGGishTransformer',
    'SimpleCNN',
    'SimpleCNN2',
    'SimpleCNN3',
    'AreaSpecificCNN',
    'AreaSpecificCNN2',
    'ASTRegressor',
    'Conv1dTransformer',
    'LeastSquareModel',
    'MultiChannelSimpleCNN'
]
SAMPLER_LIST = ['random', 'tpe']
LOSS_LIST = ['MAE', 'MSE']
TASK_PATTERN = r'range_(\d+)-(\d+)'
logger = get_logger('analysis.model_training')


def read_wav(input_folder, channel_num=1):
    """
    解析用音声データの読み込み
    Parameters
    ----------
    input_folder: str
        解析用データのフォルダパス
    channel_num: int
        読み込むチャンネル数
    Returns
    -------
    np.ndarray: [channel_num x signal_length]
        読み込んだ音声データ
    """
    with open(f'{input_folder}/signal_info.json') as f:
        signal_info = json.load(f)
    assert len(signal_info['microphone']) >= channel_num, 'channel_num must be less than microphone number'
    sig_max = signal_info['signal_max']
    result = []
    for ch in range(channel_num):
        wav_path = signal_info['info'][f'ch{ch}']['wav_path']
        if wav_path.endswith('.wav'):
            fs, signal = sp.io.wavfile.read(wav_path)
            signal *= sig_max
            if fs != FS:
                signal = librosa.resample(y=signal, orig_sr=fs, target_sr=FS)
            result.append(signal)
        elif wav_path.endswith('.flac'):
            # soundfileを使う
            signal, fs = sf.read(wav_path)
            signal *= sig_max
            if fs != FS:
                signal = librosa.resample(y=signal, orig_sr=fs, target_sr=FS)
            result.append(signal)
    signal_size = min([len(signal) for signal in result])
    result = [np.array(signal[:signal_size]) for signal in result]
    return np.array(result)


def read_crowd(input_folder, channel_num=1) -> list:
    """
    人流データ（人数）の読み込み
    Parameters
    ----------
    input_folder : str
        解析用データのフォルダパス
    channel_num : int
        読み込むチャンネル数

    Returns
    -------
    result : list of dict
        各チャンネルごとの人流データ
    """
    with open(f'{input_folder}/signal_info.json') as fp:
        signal_info = json.load(fp)
    assert len(signal_info['microphone']) >= channel_num, 'channel_num must be less than microphone number'
    # distance_list = signal_info['distance_list'] if 'distance_list' in signal_info.keys() else None
    result = []
    for ch in range(channel_num):
        crowd_path = signal_info['info'][f'ch{ch}']['each_crowd']
        df = pd.read_csv(crowd_path)
        result.append({col: df[col].tolist() for col in df.columns})
    return result


def read_grid_crowd(input_folder, x_idx_range=None, y_idx_range=None) -> dict:
    """
    グリッドごとの人流データの読み込み
    Parameters
    ----------
    input_folder: str
        解析用データのフォルダパス
    x_idx_range: tuple of int | None
        人流データのx方向インデックス範囲（(min, max)）、Noneの場合は全範囲
    y_idx_range: tuple of int | None
        人流データのy方向インデックス範囲（(min, max)）、Noneの場合は全範囲

    Returns
    -------
    result: dict
        keyはグリッド座標に対応(ex. count_x0_y0)、valueは人数の時系列データ
    """
    with open(f'{input_folder}/signal_info.json') as fp:
        signal_info = json.load(fp)
    crowd_path = signal_info['crowd_path']
    df = pd.read_csv(crowd_path)
    if x_idx_range is not None:
        df = df.loc[(df['x_idx'] >= x_idx_range[0]) & (df['x_idx'] <= x_idx_range[1])]
    if y_idx_range is not None:
        df = df.loc[(df['y_idx'] >= y_idx_range[0]) & (df['y_idx'] <= y_idx_range[1])]
    piv_df = pd.pivot_table(df, index='t', columns=['x_idx', 'y_idx'], values='count', fill_value=0)
    piv_df.columns = [f'count_x{col[0]}_y{col[1]}' for col in piv_df.columns]
    piv_df = piv_df.reset_index(drop=True)
    result = {col: piv_df[col].tolist() for col in piv_df.columns}
    return result


def calc_other_task(crowd, task_name):
    """
    task_name: range_a-b
    マイク距離ごとの人数を算出する
    """
    match = re.search(TASK_PATTERN, task_name)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        assert a < b, f'Parameter a should be less than b. Input: a = {a}, b = {b}.'
        time_length = len(crowd['count'])
        dist_a, dist_b = [0] * time_length if a == 0 else crowd[f'distance_{a}'], crowd[f'distance_{b}']
        return [db - da for da, db in zip(dist_a, dist_b)]
    else:
        Exception(f'task_name should be `range_a-b` which means number of crowds with the distance a - b.')


def read_logmel_torch(
        input_folder,
        target=None,
        tasks=None,
        channel_num=1,
        time_sec=1,
        log_scale=True,
        time_agg=False,
        astype_torch=True
):
    """
    入力フォルダの音響信号を読み込み、対数メルスペクトログラムと人流データをセットで返す

    Parameters
    ----------
    input_folder: str
        解析用データのフォルダパス
    target: list[str]
        目的変数のカラム名リスト（例: ['count']、['distance_1', 'distance_2']など）
    tasks: list[str]
        マルチタスク学習用のタスク名リスト（例: ['range_0-1', 'range_1-2']など）
    channel_num: int
        読み込むチャンネル数(モノラル限定)
    time_sec: int
        1つのサンプルに含める時間長（秒）
    log_scale: bool
        目的変数を対数変換するかどうか
    time_agg: bool
        目的変数を時間平均値にするかどうか
    astype_torch: bool
        特徴量と目的変数をtorch.Tensorで返すかどうか

    Returns
    -------
    x : torch.Tensor or np.ndarray
        抽出したlogmel特徴量、shape: (time_length, feature_dim, time_frame)
    y : torch.Tensor or np.ndarray
        抽出した人流データ、shape: (time_length, target_num)
    task : torch.Tensor or np.ndarray (optional)
        マルチタスク学習用のタスクデータ、shape: (time_length, task_num)
    """
    if target is None:
        target = ['count']
    assert channel_num == 1, 'Can input only single channel signal.'

    # wavファイル読み込み
    signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]

    # 人流カウントデータを読み込み
    crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]

    # もし目的変数カラムが存在しない場合は計算
    # （ただし人流カウントデータから計算できる場合に限る: distance_1-2みたいなカラム等）
    for tg in target:
        if tg not in crowd.keys():
            crowd[tg] = calc_other_task(crowd=crowd, task_name=tg)

    # マルチタスク学習用のデータを計算
    if tasks is not None:
        for task in tasks:
            if task not in crowd.keys():
                crowd[task] = calc_other_task(crowd=crowd, task_name=task)
    time_length = min(int(len(signal) / FS), len(crowd[target[0]]))

    # 対数メルスペクトログラム（特徴量）を計算
    x = [
        trans_logmel(signal[FS * (t + 1 - time_sec) : FS * (t + 1)], FS, astype_torch)
        for t in range(time_sec - 1, time_length)
    ]
    # torch or numpyで特徴量をxに格納
    x = torch.stack(x) if astype_torch else np.stack(x)

    # カウント値をyに格納
    y = np.array([crowd[tg][time_sec-1:time_length] for tg in target]).T

    # 目的変数を時間集約する場合
    if time_agg:
        y = np.array(
            [
                [
                    sum(crowd[tg][t + 1 - time_sec : t + 1]) / time_sec
                    for t in range(time_sec - 1, time_length)
                ]
                for tg in target
            ]
        ).T
    # 目的変数を対数化する場合
    if log_scale:
        y = np.log(y + 1.0)

    # torch型で出力する場合
    if astype_torch:
        y = torch.Tensor(y)

    # マルチタスク学習する場合
    if tasks is not None:
        task = np.array(
            [
                crowd[ts][time_sec - 1 : time_length]
                for ts in tasks
            ]
        ).T
        if time_agg:
            task = np.array(
                [
                    [
                        sum(crowd[ts][t + 1 - time_sec : t + 1]) / time_sec
                        for t in range(time_sec - 1, time_length)
                    ]
                    for ts in tasks
                ]
            ).T
        if log_scale:
            task = np.log(task + 1.0)
        if astype_torch:
            task = torch.Tensor(task)
        return x, y, task
    else:
        return x, y


def read_logmel_multichannel(
        input_folder,
        channel_num,
        x_idx_range=None,
        y_idx_range=None,
        option='torch',
        time_sec=1,
        log_scale=True,
        time_agg=True,
        astype_torch=True
):
    """
    入力フォルダ内に含まれる音響信号群と人流データを読み込み、対数メルスペクトログラムと人流ヒートマップを返す（マルチチャンネル用）
    Parameters
    ----------
    input_folder : str
        解析用データのフォルダパス
    channel_num : int
        読み込むチャンネル数
    x_idx_range : tuple of int | None
        人流データのx方向インデックス範囲（(min, max)）、Noneの場合は全範囲
    y_idx_range : tuple of int | None
        人流データのy方向インデックス範囲（(min, max)）、Noneの場合は全範囲
    option : str
        logmel特徴量の抽出方法、'VGGish' or 'torch' or 'AST'
    time_sec : int
        1つのサンプルに含める時間長（秒）
    log_scale : bool
        目的変数を対数変換するかどうか
    time_agg : bool
        目的変数を時間平均値にするかどうか
    astype_torch: bool
        特徴量と目的変数をtorch.Tensorで返すかどうか

    Returns
    -------
    x : torch.Tensor
        抽出したlogmel特徴量、shape: (time_length, channel_num, feature_dim, time_frame)
    y : torch.Tensor
        抽出した人流データ、shape: (time_length, grid_num)
    col_list : list of str
        yの各列に対応する人流データのカラム名リスト（例: count_x0_y0, count_x0_y1, ...）
    """
    # logmel特徴量の抽出方法は現状、下記のみ対応
    assert option in ['VGGish', 'torch', 'AST'], 'option must be VGGish, torch or AST.'

    # shape: (signal_length, 1)の音響信号のリストを読み込み: List[np.array]
    signal_list = read_wav(input_folder=input_folder, channel_num=channel_num)

    # 人流データ(dict形式)を読み込み、keyはグリッド座標に対応(ex. count_x0_y0)、valueは人数の時系列データ
    crowd_grid = read_grid_crowd(input_folder=input_folder, x_idx_range=x_idx_range, y_idx_range=y_idx_range)

    # 時間長を算出（一番短いものに合わせる）
    time_length = min(
        int(min([len(signal) for signal in signal_list]) / FS),  # 音響信号に関する最も短い時間長
        len(list(crowd_grid.values())[0])
    )

    x, y = None, None
    if option == 'torch':
        x = [
            [
                trans_logmel(signal[FS * (t + 1 - time_sec) : FS * (t + 1)], FS, astype_torch)
                for t in range(time_sec - 1,time_length)
            ]
            for signal in signal_list
        ]
        x = torch.stack([torch.stack(xx) for xx in x]) if astype_torch else np.stack([np.stack(xx) for xx in x])
    elif option == 'VGGish':
        logmel_func = VGGISH.get_input_processor()
        x = [
            [
                logmel_func(torch.Tensor(signal[FS * (t + 1 - time_sec):FS * (t + 1)]))
                for t in range(time_sec - 1, time_length)
            ]
            for signal in signal_list
        ]
        x = torch.stack([torch.stack(xx) for xx in x])
        if not astype_torch:
            x = x.to('cpu').detach().numpy()
    elif option == 'AST':
        raise Exception('Not implemented yet.')

    # 人流データの整形（今は2次元グリッドを1次元ベクトルで表現）
    # TODO 2次元グリッドのまま扱う場合の実装追加
    # col_list -> keys: count_x0_y0, count_x0_y1, count_x1_y0, count_x1_y1, ...
    col_list = [col for col in crowd_grid.keys() if col.startswith('count_')]
    if x_idx_range is not None and y_idx_range is not None:
        col_list = [f'count_x{xi}_y{yi}'
                    for xi in range(x_idx_range[0], x_idx_range[1]+1)
                    for yi in range(y_idx_range[0], y_idx_range[1]+1)]
    if time_agg:
        # 時間平均値を目的変数とする場合
        y = np.array([[sum(crowd_grid[cnt][t + 1 - time_sec:t + 1]) / time_sec
                          for t in range(time_sec - 1, time_length)]
                         for cnt in col_list]).T
    else:
        # 時間ごとの値を目的変数とする場合
        y = np.array([crowd_grid[cnt][time_sec - 1:time_length] for cnt in col_list]).T

    if log_scale:
        # 対数変換
        y = np.log(y + 1.0)

    if astype_torch:
        y = torch.Tensor(y)
        x = x.transpose(0, 1)
    return x, y, col_list


def read_logmel_torch_ast(input_folder, model_param, target=None, tasks=None,
                          channel_num=1, time_sec=1, log_scale=True, time_agg=False):
    # TODO 修正必要、関係する関数も全て引数から見直し
    if target is None:
        target = ['count']
    assert channel_num == 1, 'Can input only single channel signal.'
    signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]
    crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]
    for tg in target:
        if tg not in crowd.keys():
            crowd[tg] = calc_other_task(crowd=crowd, task_name=tg)
    if tasks is not None:
        for task in tasks:
            if task not in crowd.keys():
                crowd[task] = calc_other_task(crowd=crowd, task_name=task)
    time_length = min(int(len(signal) / FS), len(crowd[target[0]]))
    sig_list = np.array([signal[FS * (t+1-time_sec):FS * (t + 1)] for t in range(time_sec - 1,time_length)])
    feat_ext = ASTFeatureExtractor.from_pretrained(model_param['model_name'])
    x = feat_ext(sig_list, sampling_rate=FS, return_tensors='pt')['input_values']
    # x = [trans_logmel(signal[FS * (t+1-time_sec):FS * (t + 1)], FS) for t in range(time_sec - 1,time_length)]
    # x = torch.stack(x)
    y_np = np.array([crowd[tg][time_sec-1:time_length] for tg in target]).T
    if time_agg:
        y_np = np.array([[sum(crowd[tg][t + 1 - time_sec:t + 1]) / time_sec for t in range(time_sec - 1, time_length)]
                         for tg in target]).T
    if log_scale:
        y_np = np.log(y_np + 1.0)
    y = torch.Tensor(y_np)

    if tasks is not None:
        task_np = np.array([crowd[ts][time_sec-1:time_length] for ts in tasks]).T
        if time_agg:
            task_np = np.array([[sum(crowd[ts][t + 1 - time_sec:t + 1]) / time_sec
                                 for t in range(time_sec - 1, time_length)]
                                for ts in tasks]).T
        if log_scale:
            task_np = np.log(task_np + 1.0)
        task = torch.Tensor(task_np)
        return x, y, task
    else:
        return x, y

# def read_logmel_torch_multi_task(input_folder, target, tasks,
#                                  channel_num=1, time_sec=1, log_scale=True, time_agg=False):
#     assert channel_num == 1, 'Can input only single channel signal.'
#     signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]
#     crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]
#     for task in tasks:
#         crowd[task] = calc_other_task(crowd=crowd, task_name=task)
#     time_length = min(int(len(signal) / FS), len(crowd[tasks[0]]))
#     x = [trans_logmel(signal[FS * (t+1-time_sec):FS * (t + 1)], FS) for t in range(time_sec - 1,time_length)]
#     x = torch.stack(x)
#     y_np = np.array([crowd[tg][time_sec-1:time_length] for tg in tasks]).T
#     task_np = np.array([crowd[ts][time_sec-1:time_length] for ts in tasks]).T
#     if time_agg:
#         y_np = np.array([[sum(crowd[tg][t + 1 - time_sec:t + 1]) / time_sec for t in range(time_sec - 1, time_length)]
#                          for tg in target]).T
#         task_np = np.array([[sum(crowd[ts][t + 1 - time_sec:t + 1]) / time_sec
#                              for t in range(time_sec - 1, time_length)]
#                             for ts in tasks]).T
#     if log_scale:
#         y_np = np.log(y_np + 1.0)
#         task_np = np.log(task_np + 1.0)
#     y, task = torch.Tensor(y_np), torch.Tensor(task_np)
#     return x, y, task


def read_logmel_torch_vgg(input_folder, target=None, tasks=None,
                          channel_num=1, time_sec=1, log_scale=True, time_agg=False):
    if target is None:
        target = ['count']
    assert channel_num == 1, 'Can input only single channel signal.'
    signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]
    crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]
    for tg in target:
        if tg not in crowd.keys():
            crowd[tg] = calc_other_task(crowd=crowd, task_name=tg)
    if tasks is not None:
        for task in tasks:
            if task not in crowd.keys():
                crowd[task] = calc_other_task(crowd=crowd, task_name=task)
    logmel_func = VGGISH.get_input_processor()
    time_length = min(int(len(signal) / FS), len(crowd[target[0]]))
    # time_sec > 1の時の対応
    x = [logmel_func(torch.Tensor(signal[FS * (t + 1 - time_sec):FS * (t + 1)]))
         for t in range(time_sec - 1,time_length)]
    x = torch.stack(x)
    y_np = np.array([crowd[tg][time_sec - 1:time_length] for tg in target]).T
    if time_agg:
        y_np = np.array([[sum(crowd[tg][t + 1 - time_sec:t + 1]) / time_sec for t in range(time_sec - 1, time_length)]
                         for tg in target]).T
    if log_scale:
        y_np = np.log(y_np + 1.0)
    y = torch.Tensor(y_np)
    if tasks is not None:
        task_np = np.array([crowd[ts][time_sec-1:time_length] for ts in tasks]).T
        if time_agg:
            task_np = np.array([[sum(crowd[ts][t + 1 - time_sec:t + 1]) / time_sec
                                 for t in range(time_sec - 1, time_length)]
                                for ts in tasks]).T
        if log_scale:
            task_np = np.log(task_np + 1.0)
        task = torch.Tensor(task_np)
        return x, y, task
    else:
        return x, y


def read_logmel(model_name, input_folder, target=None, tasks=None,
                channel_num=1, time_sec=1, log_scale=True, time_agg=False, model_param=None):
    if 'VGGish' in model_name:
        return read_logmel_torch_vgg(input_folder, target, tasks, channel_num, time_sec, log_scale, time_agg)
    elif 'AST' in model_name:
        return read_logmel_torch_ast(input_folder, model_param, target, tasks, channel_num, time_sec, log_scale,
                                     time_agg)
    else:
        return read_logmel_torch(input_folder, target, tasks, channel_num, time_sec, log_scale, time_agg)


def load_dataset(input_folder_list, valid_folder_list, model_name, model_param, time_agg, log_scale, target,
                 valid_flag=True, tasks=None):
    """
    解析用データセットの読み込み
    Parameters
    ----------
    input_folder_list: list
        学習用データのフォルダリスト
    valid_folder_list: list | None
        検証用データのフォルダリスト
    model_name: str
        モデル名
    model_param: dict
        モデルのパラメータ
    time_agg: bool
        目的変数を時間平均値にするかどうか
    log_scale: bool
        目的変数を対数変換するかどうか
    target: list
        目的変数のカラム名リスト
    valid_flag: bool
        検証用データを返すかどうか
    tasks: list
        マルチタスク学習用のタスク名リスト

    Returns
    -------
    (x, y), (valid_x, valid_y) or (x, y, task), (valid_x, valid_y, valid_task)
    学習用データと検証用データ
    """
    x, y, task = None, None, None
    x_list, y_list, task_list = [], [], []
    for i, input_folder in enumerate(input_folder_list):
        logger.info(f'Reading Training Folders: {input_folder}')

        if tasks is None:
            tmp_x, tmp_y = read_logmel(
                model_name=model_name,
                input_folder=input_folder,
                target=target,
                channel_num=1,
                time_sec=model_param['time_sec'],
                time_agg=time_agg,
                log_scale=log_scale,
                model_param=model_param
            )
            x_list.append(tmp_x)
            y_list.append(tmp_y)
            # x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
            # y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)

        else:
            tmp_x, tmp_y, tmp_task = read_logmel(
                model_name=model_name,
                input_folder=input_folder,
                target=target,
                channel_num=1,
                tasks=tasks,
                time_sec=model_param['time_sec'],
                time_agg=time_agg,
                log_scale=log_scale
            )
            x_list.append(tmp_x)
            y_list.append(tmp_y)
            task_list.append(tmp_task)
            # x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
            # y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)
            # task = tmp_task if i == 0 else torch.cat([task, tmp_task], dim=0)

    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    if tasks is not None:
        task = torch.cat(task_list, dim=0)

    # 検証用のデータがいらない場合 (予測のみ行う場合)
    if not valid_flag:
        if tasks is None:
            return x, y
        else:
            return x, y, task

    valid_x, valid_y, valid_task = None, None, None
    valid_x_list, valid_y_list, valid_task_list = [], [], []
    # valid_folder_listがNoneの時はinput_folder_listのデータをランダムで学習用、評価用に分割する
    if valid_folder_list is None:
        tr_idx, ts_idx = train_test_split(range(len(y)), test_size=0.2, random_state=0)
        valid_x, valid_y = x[ts_idx], y[ts_idx]
        x, y = x[tr_idx], y[tr_idx]

        if tasks is not None:
            valid_task = task[ts_idx]
            task = task[tr_idx]

    else:
        for i, valid_folder in enumerate(valid_folder_list):
            logger.info(f'Reading Valid Folders: {valid_folder}')

            if tasks is None:
                tmp_x, tmp_y = read_logmel(
                    model_name=model_name,
                    input_folder=valid_folder,
                    target=target,
                    channel_num=1,
                    time_sec=model_param['time_sec'],
                    time_agg=time_agg,
                    model_param=model_param
                )
                valid_x_list.append(tmp_x)
                valid_y_list.append(tmp_y)
                # valid_x = tmp_x if i == 0 else torch.cat([valid_x, tmp_x], dim=0)
                # valid_y = tmp_y if i == 0 else torch.cat([valid_y, tmp_y], dim=0)

            else:
                tmp_x, tmp_y, tmp_task = read_logmel(
                    model_name=model_name,
                    input_folder=valid_folder,
                    target=target,
                    channel_num=1,
                    tasks=tasks,
                    time_sec=model_param['time_sec'],
                    time_agg=time_agg
                )
                valid_x_list.append(tmp_x)
                valid_y_list.append(tmp_y)
                valid_task_list.append(tmp_task)
                # valid_x = tmp_x if i == 0 else torch.cat([valid_x, tmp_x], dim=0)
                # valid_y = tmp_y if i == 0 else torch.cat([valid_y, tmp_y], dim=0)
                # valid_task = tmp_task if i == 0 else torch.cat([valid_task, tmp_task], dim=0)

        valid_x = torch.cat(valid_x_list, dim=0)
        valid_y = torch.cat(valid_y_list, dim=0)
        if tasks is not None:
            valid_task = torch.cat(valid_task_list, dim=0)

    if tasks is None:
        return (x, y), (valid_x, valid_y)
    else:
        return (x, y, task), (valid_x, valid_y, valid_task)


def load_dataset_multichannel(input_folder_list, valid_folder_list, model_name, model_param,
                              x_idx_range=None, y_idx_range=None,
                              time_agg=True, log_scale=True, valid_flag=True):
    # TODO AST未対応
    option = 'VGGish' if 'VGGish' in model_name else 'torch'

    # x, y, col_list = None, None, None
    col_list = None
    # input_folder_listのデータを読み込み
    x_list, y_list = [], []
    for i, input_folder in enumerate(input_folder_list):
        logger.info(f'Reading Training Folders: {input_folder}')
        tmp_x, tmp_y, col_list = read_logmel_multichannel(
            input_folder=input_folder,
            channel_num=model_param['channel_num'],
            x_idx_range=x_idx_range,
            y_idx_range=y_idx_range,
            option=option,
            time_sec=model_param['time_sec'],
            time_agg=time_agg,
            log_scale=log_scale
        )
        x_list.append(tmp_x)
        y_list.append(tmp_y)
        # x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
        # y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)

    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    if not valid_flag:
        return (x, y), col_list

    # valid_x, valid_y = None, None
    # valid_folder_listがNoneの時はinput_folder_listのデータをランダムで学習用、評価用に分割する
    if valid_folder_list is None:
        tr_idx, ts_idx = train_test_split(range(len(y)), test_size=0.2, random_state=0)
        valid_x, valid_y = x[ts_idx], y[ts_idx]
        x, y = x[tr_idx], y[tr_idx]

    else:
        valid_x_list, valid_y_list = [], []
        for i, valid_folder in enumerate(valid_folder_list):
            logger.info(f'Reading Valid Folders: {valid_folder}')

            tmp_x, tmp_y, tmp_col = read_logmel_multichannel(
                input_folder=valid_folder,
                channel_num=model_param['channel_num'],
                x_idx_range=x_idx_range,
                y_idx_range=y_idx_range,
                option=option,
                time_sec=model_param['time_sec'],
                time_agg=time_agg,
                log_scale=log_scale
            )
            valid_x_list.append(tmp_x)
            valid_y_list.append(tmp_y)
            # valid_x = tmp_x if i == 0 else torch.cat([valid_x, tmp_x], dim=0)
            # valid_y = tmp_y if i == 0 else torch.cat([valid_y, tmp_y], dim=0)
        valid_x = torch.cat(valid_x_list, dim=0)
        valid_y = torch.cat(valid_y_list, dim=0)

    return (x, y), (valid_x, valid_y), col_list


def audio_crowd_model(model_name, model_param):
    """
    model_nameからモデルを取得し、model_paramを引数としてモデルを生成する
    Parameters
    ----------
    model_name: str
        モデル名
    model_param: dict
        モデルのパラメータ

    Returns
    -------
    model: nn.Module
        生成したモデル
    """
    assert model_name in MODEL_LIST, f'You can choose model as following: {MODEL_LIST}.'
    # model_nameからモデルの関数を取得
    func = eval(model_name)

    # 当該モデルの引数を取得してmodel_paramから該当する引数のみ渡す
    func_params = list(signature(func).parameters.keys())
    setting_params = {k: model_param[k] for k in func_params if k in model_param.keys()}
    return func(**setting_params)


def trial_from_model_param_setting(trial: optuna.Trial, model_param_setting: dict):
    """
    optunaのtrialオブジェクトからモデルパラメタを取得する
    Parameters
    ----------
    trial: optunaのtrialオブジェクト
    model_param_setting: dict
        optunaに設定する際に入力したJSONデータ

    Returns
    -------
    model_param_optuna: dict
        optuna探索により得られたモデルパラメタを格納したJSONデータ
    """
    model_param_optuna = {}
    for name, val in model_param_setting.items():
        if not isinstance(val, dict):
            model_param_optuna[name] = val
        else:
            if val['type'].count('list') == 2:
                assert isinstance(val['size'], list), f'double list setting: list_size should be list'
                list_size = [
                    val['size'][0] if isinstance(val['size'][0], int) else model_param_optuna[val['size'][0]],
                    val['size'][1] if isinstance(val['size'][1], int) else model_param_optuna[val['size'][1]]
                ]
                if 'int' in val['type']:
                    model_param_optuna[name] = [
                        [
                            trial.suggest_int(name=f'{name}_{i0}_{i1}',
                                              low=val['low'],
                                              high=val['high'])
                            for i1 in range(list_size[1])
                        ]
                        for i0 in range(list_size[0])
                    ]
                elif 'float' in val['type']:
                    model_param_optuna[name] = [
                        [
                            trial.suggest_float(name=f'{name}_{i0}_{i1}',
                                                low=val['low'],
                                                high=val['high'])
                            for i1 in range(list_size[1])
                        ]
                        for i0 in range(list_size[0])
                    ]
                elif 'category' in val['type']:
                    model_param_optuna[name] = [
                        [
                            trial.suggest_categorical(name=f'{name}_{i0}_{i1}',
                                                      choices=val['category'])
                            for i1 in range(list_size[1])
                        ]
                        for i0 in range(list_size[0])
                    ]
                else:
                    Exception(f'Invalid model parameter setting, {name}: {val}')

            elif val['type'].count('list') == 1:
                list_size = val['size'] if isinstance(val['size'], int) else model_param_optuna[val['size']]
                if 'int' in val['type']:
                    model_param_optuna[name] = [trial.suggest_int(name=f'{name}_{idx}',
                                                                  low=val['low'],
                                                                  high=val['high'])
                                                for idx in range(list_size)]
                elif 'float' in val['type']:
                    model_param_optuna[name] = [trial.suggest_float(name=f'{name}_{idx}',
                                                                    low=val['low'],
                                                                    high=val['high'])
                                                for idx in range(list_size)]
                elif 'category' in val['type']:
                    model_param_optuna[name] = [trial.suggest_categorical(name=f'{name}_{idx}',
                                                                          choices=val['category'])
                                                for idx in range(list_size)]
                else:
                    Exception(f'Invalid model parameter setting, {name}: {val}')

            elif val['type'].count('list') == 0:
                if val['type'] == 'int':
                    model_param_optuna[name] = trial.suggest_int(name=name, low=val['low'], high=val['high'])
                elif val['type'] == 'float':
                    model_param_optuna[name] = trial.suggest_float(name=name, low=val['low'], high=val['high'])
                elif val['type'] == 'category':
                    model_param_optuna[name] = trial.suggest_categorical(name=name, choices=val['category'])
                else:
                    Exception(f'Invalid model parameter setting, {name}: {val}')

            else:
                cnt = val['type'].count('list')
                Exception(f'Invalid type: {cnt} layers list does not work for now.')

    return model_param_optuna


def model_param_from_best_trial(model_param_setting: dict, best_params_optuna: dict):
    """
    optuna探索結果から最良となったパラメタについて、dict化(listなど)
    Parameters
    ----------
    model_param_setting: dict
        optunaに設定する際に入力したJSONデータ
    best_params_optuna: dict
        optuna探索により得られた最良の結果

    Returns
    -------
    best_params: dict
        探索結果のモデルパラメタを格納したJSONデータ
    """
    best_params = {}
    for name, val in model_param_setting.items():
        if not isinstance(val, dict):
            # 探索時の設定ファイル
            best_params[name] = val
        else:
            if val['type'].count('list') == 2:
                assert isinstance(val['size'], list), f'double list setting: list_size should be list'
                list_size = []
                for val_size in val['size']:
                    if isinstance(val_size, str):
                        if isinstance(model_param_setting[val_size], int):
                            # 要素数が固定値でなく、別パラメータを参照しており、またoptunaで探索させていない場合
                            list_size.append(model_param_setting[val_size])
                        else:
                            # optunaで要素数を探索させている場合はoptunaの結果から取得
                            list_size.append(best_params_optuna[val_size])
                    else:
                        # 要素数が固定値の場合
                        list_size.append(val_size)
                best_params[name] = [
                    [
                        best_params_optuna[f'{name}_{i0}_{i1}']
                        for i1 in range(list_size[1])
                    ]
                    for i0 in range(list_size[0])
                ]

            elif val['type'].count('list') == 1:
                if isinstance(val['size'], str):
                    if isinstance(model_param_setting[val['size']], int):
                        # 要素数が固定値でなく、別パラメータを参照しており、またoptunaで探索させていない場合
                        list_size = model_param_setting[val['size']]
                    else:
                        # optunaで要素数を探索させている場合はoptunaの結果から取得
                        list_size = best_params_optuna[val['size']]
                else:
                    # 要素数が固定値の場合
                    list_size = val['size']
                best_params[name] = [best_params_optuna[f'{name}_{idx}'] for idx in range(list_size)]

            elif val['type'].count('list') == 0:
                best_params[name] = best_params_optuna[name]

            else:
                Exception('3重以上のlistにはまだ未対応')
            # if 'list' in val['type']:
            #     if isinstance(val['size'], str):
            #         if isinstance(model_param_setting[val['size']], int):
            #             # 要素数が固定値でなく、別パラメータを参照しており、またoptunaで探索させていない場合
            #             list_size = model_param_setting[val['size']]
            #         else:
            #             # optunaで要素数を探索させている場合はoptunaの結果から取得
            #             list_size = best_params_optuna[val['size']]
            #     else:
            #         # 要素数が固定値の場合
            #         list_size = val['size']
            #     best_params[name] = [best_params_optuna[f'{name}_{idx}'] for idx in range(list_size)]
            # else:
            #     best_params[name] = best_params_optuna[name]
    return best_params


def is_valid_model(model_name, model_param):
    """
    model_paramでモデルを設定した際に、そのモデルが問題ないか検証する

    Parameters
    ----------
    model_name: str
        モデル名：現状SimpleCNN2, MultiTaskCNNのみ検証の必要がある
    model_param: dict
        モデルパラメタ

    Returns
    -------
    valid: bool
        モデルパラメタが妥当であればTrue、そうでなければFalse
    """
    assert model_name in MODEL_LIST, f'You can choose model: {MODEL_LIST}.'
    if model_name in ['SimpleCNN2', 'SimpleCNN3', 'AreaSpecificCNN', 'AreaSpecificCNN2', 'MultiChannelSimpleCNN']:
        func = eval(model_name)
        func_params = list(signature(func.is_valid).parameters.keys())
        setting_params = {k: model_param[k] for k in func_params if k in model_param.keys()}
        return func.is_valid(**setting_params)
    # if model_name == 'SimpleCNN2':
    #     return SimpleCNN2.is_valid(
    #         frame_num=model_param['frame_num'],
    #         freq_num=model_param['freq_num'],
    #         kernel_size=model_param['kernel_size'],
    #         dilation_size=model_param['dilation_size'],
    #         layer_num=model_param['layer_num'],
    #         inter_ch=model_param['inter_ch']
    #     )
    # elif model_name == 'AreaSpecificCNN':
    #     return AreaSpecificCNN.is_valid(
    #         task_num=model_param['task_num'],
    #         freq_num=model_param['task_num'],
    #         frame_num=model_param['frame_num'],
    #         kernel_size=model_param['kernel_size'],
    #         dilation_size=model_param['dilation_size'],
    #         layer_num=model_param['layer_num'],
    #         inter_ch=model_param['inter_ch'],
    #         pool_size=model_param['pool_size']
    #     )
    # elif model_name == 'AreaSpecificCNN2':
    #     return AreaSpecificCNN2.is_valid(
    #         task_num=model_param['task_num'],
    #         freq_num=model_param['freq_num'],
    #         frame_num=model_param['frame_num'],
    #         common_kernel_size=model_param['common_kernel_size'],
    #         kernel_size=model_param['kernel_size'],
    #         common_dilation_size=model_param['common_dilation_size'],
    #         dilation_size=model_param['dilation_size'],
    #         common_layer_num=model_param['common_layer_num'],
    #         layer_num=model_param['layer_num'],
    #         common_inter_ch=model_param['common_inter_ch'],
    #         inter_ch=model_param['inter_ch'],
    #         common_pool_size=model_param['common_pool_size'],
    #         pool_size=model_param['pool_size']
    #     )
    return True


def load_model_setting(model_name, model_param, target_num, finetune=None):
    """
    モデル、オプティマイザ、損失関数の生成
    Parameters
    ----------
    model_name: str
        モデル名
    model_param: dict
        モデルのパラメータ
    target_num: int
        目的変数の数
    finetune: any
        ファインチューニング用のパラメータ
        Noneの時は全てのパラメータを学習する

    Returns
    -------
    model: nn.Module
        生成したモデル
    optimizer: torch.optim.Optimizer
        生成したオプティマイザ
    criterion: nn.Module
        生成した損失関数
    """
    if 'out_features' not in model_param:
        model_param['out_features'] = target_num
    model = audio_crowd_model(model_name, model_param)
    if finetune is not None:
        model.freeze(finetune)

    # lr = model_param.get('lr', 1e-5)
    # weight_decay = 0.0 if 'weight_decay' not in model_param.keys() else model_param['weight_decay']
    optimizer = torch.optim.Adam(
        model.parameters() if finetune is None else filter(lambda p: p.requires_grad, model.parameters()),
        lr=model_param.get('lr', 1e-5),
        weight_decay=model_param.get('weight_decay', 0.0)
    )
    criterion = nn.MSELoss()
    if 'criterion' in model_param.keys():
        assert model_param['criterion'] in LOSS_LIST
        criterion = nn.L1Loss() if model_param['criterion'] == 'MAE' else nn.MSELoss()
    return model, optimizer, criterion


def audio_crowd_training(input_folder_list, valid_folder_list,
                         model_folder, model_name, model_param, target, epoch, dev=None,
                         log_scale=True, time_agg=False, batch_size=64):
    device = get_device(dev)
    logger.info(f'Start {model_name} Training: {device} - Columns: {target}')
    logger.info(f' model_folder: {model_folder}')
    (x, y), (valid_x, valid_y) = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        log_scale=log_scale,
        target=target
    )

    train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
    test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model, optimizer, criterion = load_model_setting(model_name, model_param, len(target))
    model = model.to(device)

    train_loss, test_loss = [], []
    for ep in range(epoch):
        tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, ep)
        ts_loss_tmp = model_test(model, test_dataloader, criterion, ep)
        train_loss.append(tr_loss_tmp)
        test_loss.append(ts_loss_tmp)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)
    view_loss(train_loss, test_loss, f'{model_folder}/loss.png')
    torch.save(model.state_dict(), f'{model_folder}/{model_name}_model.pt')

    target_np, output_np = model_predict(model, train_dataloader)
    plot_result(model_folder, target_np, output_np, target, label='train', log_scale=log_scale)
    write_result(model_folder, target_np, output_np, target, label='train', log_scale=log_scale)

    target_np, output_np = model_predict(model, test_dataloader)
    plot_result(model_folder, target_np, output_np, target, label='test', log_scale=log_scale)
    write_result(model_folder, target_np, output_np, target, label='test', log_scale=log_scale)
    return


def audio_crowd_training_multichannel(
        input_folder_list: list[str],
        valid_folder_list: list[str],
        model_folder: str,
        model_name: str,
        model_param: dict,
        epoch: int,
        dev=None,
        log_scale=True,
        time_agg=True,
        batch_size=64
):
    device = get_device(dev)
    logger.info(f'Start {model_name} Training: {device}')
    logger.info(f' model_folder: {model_folder}')
    (x, y), (valid_x, valid_y), col_list = load_dataset_multichannel(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        log_scale=log_scale
    )
    # train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
    # test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device))
    train_dataset = torch.utils.data.TensorDataset(x, y)
    test_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    # サンプル重みの計算インスタンス
    swc = SampleWeightCalculator(y=y, dev=device, alpha=0.1)

    model, optimizer, criterion = load_model_setting(model_name, model_param, len(col_list))
    model = model.to(device)

    train_loss, test_loss = [], []
    for ep in range(epoch):
        # tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, ep, device)
        # ts_loss_tmp = model_test(model, test_dataloader, criterion, ep, device)
        tr_loss_tmp = model_train_sw(model, train_dataloader, criterion, optimizer, ep, swc, device)
        ts_loss_tmp = model_test_sw(model, test_dataloader, criterion, ep, swc, device)
        train_loss.append(tr_loss_tmp)
        test_loss.append(ts_loss_tmp)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)
    view_loss(train_loss, test_loss, f'{model_folder}/loss.png')
    torch.save(model.state_dict(), f'{model_folder}/{model_name}_model.pt')

    target_np, output_np = model_predict(model, train_dataloader, dev)
    plot_result(model_folder, target_np, output_np, col_list, label='train', log_scale=log_scale)
    write_result(model_folder, target_np, output_np, col_list, label='train', log_scale=log_scale)

    target_np, output_np = model_predict(model, test_dataloader, dev)
    plot_result(model_folder, target_np, output_np, col_list, label='test', log_scale=log_scale)
    write_result(model_folder, target_np, output_np, col_list, label='test', log_scale=log_scale)
    return


def audio_crowd_training_multitask(input_folder_list, valid_folder_list,
                                   model_folder, model_name, model_param, target, epoch, weight, tasks=None, dev=None,
                                   log_scale=True, time_agg=False, batch_size=64):
    device = get_device(dev)
    logger.info(f'Start {model_name} Training: {device} - Columns: {target}')
    (x, y, task), (valid_x, valid_y, valid_task) = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        target=target,
        log_scale=log_scale,
        tasks=tasks
    )
    # x, y, task = None, None, None
    # for i, input_folder in enumerate(input_folder_list):
    #     logger.info(f'Reading Training Folders: {input_folder}')
    #     tmp_x, tmp_y, tmp_task = read_logmel(model_name=model_name, input_folder=input_folder, target=target,
    #                                          channel_num=1, tasks=tasks, time_sec=model_param['time_sec'],
    #                                          time_agg=time_agg)
    #     x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
    #     y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)
    #     task = tmp_task if i == 0 else torch.cat([task, tmp_task], dim=0)
    #
    # valid_x, valid_y, valid_task = None, None, None
    # if valid_folder_list is None:
    #     tr_idx, ts_idx = train_test_split(range(len(y)), test_size=0.2, random_state=0)
    #     valid_x, valid_y, valid_task = x[ts_idx], y[ts_idx], task[ts_idx]
    #     x, y, task = x[tr_idx], y[tr_idx], task[tr_idx]
    # else:
    #     for i, valid_folder in enumerate(valid_folder_list):
    #         logger.info(f'Reading Valid Folders: {valid_folder}')
    #         tmp_x, tmp_y, tmp_task = read_logmel(model_name=model_name, input_folder=valid_folder, target=target,
    #                                              channel_num=1, tasks=tasks, time_sec=model_param['time_sec'],
    #                                              time_agg=time_agg)
    #         valid_x = tmp_x if i == 0 else torch.cat([valid_x, tmp_x], dim=0)
    #         valid_y = tmp_y if i == 0 else torch.cat([valid_y, tmp_y], dim=0)
    #         valid_task = tmp_task if i == 0 else torch.cat([valid_task, tmp_task], dim=0)

    train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device), task.to(device))
    test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device), valid_task.to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    if 'out_features' not in model_param:
        model_param['out_features'] = len(target)
    model = audio_crowd_model(model_name, model_param).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=2.7e-09)
    criterion = nn.MSELoss()
    task_criterion = nn.MSELoss()

    train_loss, test_loss = [], []
    main_train_loss, main_test_loss = [], []
    task_train_loss, task_test_loss = [], []
    for ep in range(epoch):
        tr_loss_tmp = model_train_multitask(model, train_dataloader, criterion, task_criterion, optimizer, ep,
                                            weight=weight, verbose=True)
        ts_loss_tmp = model_test_multitask(model, test_dataloader, criterion, task_criterion, ep,
                                           weight=weight, verbose=True)
        train_loss.append(tr_loss_tmp[0])
        main_train_loss.append(tr_loss_tmp[1])
        task_train_loss.append(tr_loss_tmp[2])

        test_loss.append(ts_loss_tmp[0])
        main_test_loss.append(ts_loss_tmp[1])
        task_test_loss.append(ts_loss_tmp[2])

    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)
    view_loss(train_loss, test_loss, f'{model_folder}/loss.png')
    view_multi_loss(multi_loss_list=[train_loss, main_train_loss, task_train_loss],
                    label_list=['Total', 'Main', 'Sub-task'],
                    filename=f'{model_folder}/train_loss.png')
    view_multi_loss(multi_loss_list=[test_loss, main_test_loss, task_test_loss],
                    label_list=['Total', 'Main', 'Sub-task'],
                    filename=f'{model_folder}/test_loss.png')

    torch.save(model.state_dict(), f'{model_folder}/{model_name}_model.pt')

    target_np, output_np, target_task_np, output_task_np = model_predict_multitask(model, train_dataloader)
    if log_scale:
        scatter_plot(target_np, output_np, f'{model_folder}/train_scatter_log.png')
        for i, tsk in enumerate(tasks):
            scatter_plot(target_task_np[:, i], output_task_np[: ,i],
                         f'{model_folder}/train_scatter_log_{tsk}.png')
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
        target_task_np = np.exp(target_task_np) - 1
        output_task_np = np.exp(output_task_np) - 1
    scatter_plot(target_np, output_np, f'{model_folder}/train_scatter.png')
    for i, tsk in enumerate(tasks):
        scatter_plot(target_task_np[:, i], output_task_np[:, i],
                     f'{model_folder}/train_scatter_{tsk}.png')

    target_np, output_np, target_task_np, output_task_np = model_predict_multitask(model, test_dataloader)
    if log_scale:
        scatter_plot(target_np, output_np, f'{model_folder}/scatter_log.png')
        for i, tsk in enumerate(tasks):
            scatter_plot(target_task_np[:, i], output_task_np[:, i],
                         f'{model_folder}/scatter_log_{tsk}.png')
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
        target_task_np = np.exp(target_task_np) - 1
        output_task_np = np.exp(output_task_np) - 1
    scatter_plot(target_np, output_np, f'{model_folder}/scatter.png')
    write_result(model_folder, target_np, output_np, label=target[0])
    for i, tsk in enumerate(tasks):
        scatter_plot(target_task_np[:, i], output_task_np[:, i],
                     f'{model_folder}/scatter_{tsk}.png')
        write_result(model_folder, target_task_np[:, i], output_task_np[:, i], label=tsk)

    return


def audio_crowd_prediction(input_folder_list, model_folder, model_name, output_folder, model_param, target,
                           dev=None, log_scale=True, time_agg=False, batch_size=64):
    device = get_device(dev)
    logger.info(f'Start {model_name} Prediction: {device} - Columns: {target}')
    x, y = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=None,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        log_scale=log_scale,
        target=target,
        valid_flag=False
    )
    # x, y = None, None
    # for i, input_folder in enumerate(input_folder_list):
    #     logger.info(f'Reading Folders: {input_folder}')
    #     tmp_x, tmp_y = read_logmel(model_name=model_name, input_folder=input_folder, target=target, channel_num=1,
    #                                time_sec=model_param['time_sec'], time_agg=time_agg)
    #     x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
    #     y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)
    if 'out_features' not in model_param:
        model_param['out_features'] = len(target)
    model = audio_crowd_model(model_name, model_param)

    model_param = torch.load(f'{model_folder}/{model_name}_model.pt')
    model.load_state_dict(model_param)
    model = model.to(device)
    test_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    target_np, output_np = model_predict(model, test_dataloader)

    # TODO input_folder_listの入力wavファイルごとの出力
    os.makedirs(output_folder, exist_ok=True)
    if log_scale:
        scatter_plot(target_np, output_np, f'{output_folder}/scatter_log.png')
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
    scatter_plot(target_np, output_np, f'{output_folder}/scatter.png')
    target_df = pd.DataFrame(target_np)
    target_df.columns = [f'target{i}' for i in range(len(target_df.columns))]
    output_df = pd.DataFrame(output_np)
    output_df.columns = [f'predict{i}' for i in range(len(output_df.columns))]
    res_df = pd.concat([target_df, output_df], axis=1)
    res_df.to_csv(f'{output_folder}/result.csv', index=False)
    calculate_accuracy(target_np, output_np, f'{output_folder}/acc.json')
    return


def audio_crowd_tuning(input_folder_list, valid_folder_list,
                       model_folder, model_name, model_param, target, epoch, n_trials=1000, sampler='tpe',
                       dev=None, log_scale=True, time_agg=False, batch_size=64):
    device = get_device(dev)
    logger.info(f'Start {model_name} Training: {device} - Columns: {target}')
    logger.info(f' model_folder: {model_folder}')
    (x, y), (valid_x, valid_y) = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        log_scale=log_scale,
        target=target
    )

    train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
    test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    if 'out_features' not in model_param:
        model_param['out_features'] = len(target)
    # model = audio_crowd_model(model_name, model_param).to(device)

    def objective(trial: optuna.Trial):
        model_param_optuna = trial_from_model_param_setting(trial, model_param)
        if not is_valid_model(model_name, model_param_optuna):
            # このtrialは無効としてskip
            return float('inf')

        model = audio_crowd_model(model_name, model_param_optuna).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=model_param.get('lr', 1e-5),
                                     weight_decay=model_param.get('weight_decay', 0.0))
        criterion = nn.MSELoss()
        if 'criterion' in model_param.keys():
            assert model_param['criterion'] in LOSS_LIST
            criterion = nn.L1Loss() if model_param['criterion'] == 'MAE' else nn.MSELoss()

        train_loss, test_loss = [], []
        for ep in range(epoch):
            tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, ep, verbose=True)
            ts_loss_tmp = model_test(model, test_dataloader, criterion, ep, verbose=True)
            train_loss.append(tr_loss_tmp)
            test_loss.append(ts_loss_tmp)

        each_folder = f'{model_folder}/trial{trial._trial_id}'
        if not os.path.exists(each_folder):
            os.makedirs(each_folder, exist_ok=True)
        view_loss(train_loss, test_loss, f'{each_folder}/loss.png')

        target_np, output_np = model_predict(model, train_dataloader, verbose=False)
        plot_result(each_folder, target_np, output_np, target, label='train', log_scale=log_scale)
        write_result(each_folder, target_np, output_np, target, label='train', log_scale=log_scale)

        target_np, output_np = model_predict(model, test_dataloader, verbose=False)
        plot_result(each_folder, target_np, output_np, target, label='test', log_scale=log_scale)
        write_result(each_folder, target_np, output_np, target, label='test', log_scale=log_scale)

        with open(f'{each_folder}/model_params.json', 'w') as fp_:
            json.dump(model_param_optuna, fp_, indent=4)

        return test_loss[-1]

    optuna_sampler = optuna.samplers.TPESampler(seed=42) if sampler == 'tpe' else optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna_sampler,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = model_param_from_best_trial(model_param_setting=model_param,
                                              best_params_optuna=study.best_params)
    with open(f'{model_folder}/best_params.json', 'w') as fp:
        json.dump(best_params, fp, indent=4)
    df = study.trials_dataframe()
    df.to_csv(f'{model_folder}/optuna_trials.csv', index=False)
    return


def audio_crowd_finetune(input_folder_list, valid_folder_list, pretrain_model_folder,
                         model_folder, model_name, model_param, target, epoch, dev=None,
                         log_scale=True, time_agg=False, batch_size=64):
    device = get_device(dev)
    logger.info(f'Start {model_name} Finetuning: {device} - Columns: {target}')
    (x, y), (valid_x, valid_y) = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        log_scale=log_scale,
        target=target
    )
    train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
    test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model, optimizer, criterion = load_model_setting(model_name, model_param, len(target),
                                                     finetune=model_param['finetune'])
    model = model.to(device)
    # Load pre-trained model
    model.load_state_dict(torch.load(f'{pretrain_model_folder}/{model_name}_model.pt'))

    train_loss, test_loss = [], []
    for ep in range(epoch):
        tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, ep)
        ts_loss_tmp = model_test(model, test_dataloader, criterion, ep)
        train_loss.append(tr_loss_tmp)
        test_loss.append(ts_loss_tmp)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)
    view_loss(train_loss, test_loss, f'{model_folder}/loss.png')
    torch.save(model.state_dict(), f'{model_folder}/{model_name}_model.pt')

    target_np, output_np = model_predict(model, train_dataloader)
    plot_result(model_folder, target_np, output_np, target, label='train', log_scale=log_scale)
    write_result(model_folder, target_np, output_np, target, label='train', log_scale=log_scale)

    target_np, output_np = model_predict(model, test_dataloader)
    plot_result(model_folder, target_np, output_np, target, label='test', log_scale=log_scale)
    write_result(model_folder, target_np, output_np, target, label='test', log_scale=log_scale)
    return


def audio_crowd_tuning_multitask(input_folder_list, valid_folder_list,
                                 model_folder, model_name, model_param, target, tasks, epoch, weight, n_trials=1000,
                                 sampler='tpe', dev=None, log_scale=True, time_agg=False, batch_size=64):
    device = get_device(dev)
    logger.info(f'Start {model_name} Training: {device} - Columns: {target}')
    (x, y, task), (valid_x, valid_y, valid_task) = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        log_scale=log_scale,
        target=target,
        tasks=tasks
    )

    train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device), task.to(device))
    test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device), valid_task.to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    if 'out_features' not in model_param:
        model_param['out_features'] = len(target)

    def objective(trial: optuna.Trial):
        model_param_optuna = trial_from_model_param_setting(trial, model_param)

        if not is_valid_model(model_name, model_param_optuna):
            return float('inf')

        model = audio_crowd_model(model_name, model_param_optuna).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_param.get('lr', 1e-5),
                                     weight_decay=model_param.get('weight_decay', 0.0))
        criterion = nn.MSELoss()
        task_criterion = nn.MSELoss()
        if 'criterion' in model_param.keys():
            assert model_param['criterion'] in LOSS_LIST
            criterion = nn.L1Loss() if model_param['criterion'] == 'MAE' else nn.MSELoss()
            task_criterion = nn.L1Loss() if model_param['criterion'] == 'MAE' else nn.MSELoss()

        train_loss, test_loss = [], []
        main_train_loss, main_test_loss = [], []
        task_train_loss, task_test_loss = [], []
        for ep in range(epoch):
            tr_loss_tmp = model_train_multitask(model, train_dataloader, criterion, task_criterion, optimizer, ep,
                                                weight=weight, verbose=False)
            ts_loss_tmp = model_test_multitask(model, test_dataloader, criterion, task_criterion, ep,
                                               weight=weight, verbose=False)
            train_loss.append(tr_loss_tmp[0])
            main_train_loss.append(tr_loss_tmp[1])
            task_train_loss.append(tr_loss_tmp[2])

            test_loss.append(ts_loss_tmp[0])
            main_test_loss.append(ts_loss_tmp[1])
            task_test_loss.append(ts_loss_tmp[2])

        each_folder = f'{model_folder}/trial{trial._trial_id}'
        if not os.path.exists(each_folder):
            os.makedirs(each_folder, exist_ok=True)
        view_loss(train_loss, test_loss, f'{each_folder}/loss.png')
        view_multi_loss(multi_loss_list=[train_loss, main_train_loss, task_train_loss],
                        label_list=['Total', 'Main', 'Sub-task'],
                        filename=f'{each_folder}/train_loss.png')
        view_multi_loss(multi_loss_list=[test_loss, main_test_loss, task_test_loss],
                        label_list=['Total', 'Main', 'Sub-task'],
                        filename=f'{each_folder}/test_loss.png')

        target_np, output_np, target_task_np, output_task_np = model_predict_multitask(model, train_dataloader,
                                                                                       verbose=False)
        if log_scale:
            scatter_plot(target_np, output_np, f'{each_folder}/train_scatter_log.png')
            for i, tsk in enumerate(tasks):
                scatter_plot(target_task_np[:, i], output_task_np[:, i],
                             f'{each_folder}/train_scatter_log_{tsk}.png')
            target_np = np.exp(target_np) - 1
            output_np = np.exp(output_np) - 1
            target_task_np = np.exp(target_task_np) - 1
            output_task_np = np.exp(output_task_np) - 1
        scatter_plot(target_np, output_np, f'{each_folder}/train_scatter.png')
        for i, tsk in enumerate(tasks):
            scatter_plot(target_task_np[:, i], output_task_np[:, i],
                         f'{each_folder}/train_scatter_{tsk}.png')

        target_np, output_np, target_task_np, output_task_np = model_predict_multitask(model, test_dataloader,
                                                                                       verbose=False)
        if log_scale:
            scatter_plot(target_np, output_np, f'{each_folder}/scatter_log.png')
            for i, tsk in enumerate(tasks):
                scatter_plot(target_task_np[:, i], output_task_np[:, i],
                             f'{each_folder}/scatter_log_{tsk}.png')
            target_np = np.exp(target_np) - 1
            output_np = np.exp(output_np) - 1
            target_task_np = np.exp(target_task_np) - 1
            output_task_np = np.exp(output_task_np) - 1
        scatter_plot(target_np, output_np, f'{each_folder}/scatter.png')
        for i, tsk in enumerate(tasks):
            scatter_plot(target_task_np[:, i], output_task_np[:, i],
                         f'{each_folder}/scatter_{tsk}.png')

        # tried_params = model_param_from_best_trial(model_param, model_param_optuna)
        with open(f'{each_folder}/model_params.json', 'w') as fp_:
            json.dump(model_param_optuna, fp_)
        return test_loss[-1]

    optuna_sampler = optuna.samplers.TPESampler(seed=42) if sampler == 'tpe' else optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna_sampler,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = model_param_from_best_trial(model_param, study.best_params)
    with open(f'{model_folder}/best_params.json', 'w') as fp:
        json.dump(best_params, fp)
    df = study.trials_dataframe()
    df.to_csv(f'{model_folder}/optuna_trials.csv', index=False)
    return


def audio_crowd_tuning_multichannel(
        input_folder_list,
        valid_folder_list,
        model_folder,
        model_name,
        model_param,
        epoch,
        n_trials=1000,
        sampler='tpe',
        dev=None,
        log_scale=True,
        time_agg=True,
        batch_size=64
):
    device = get_device(dev)
    logger.info(f'Start {model_name} Tuning: {device}')
    logger.info(f' model_folder: {model_folder}')
    (x, y), (valid_x, valid_y), col_list = load_dataset_multichannel(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        log_scale=log_scale
    )
    # train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
    # test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device))
    train_dataset = torch.utils.data.TensorDataset(x, y)
    test_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    swc = SampleWeightCalculator(y=y, dev=device, alpha=0.1)

    def objective(trial: optuna.Trial):
        model_param_optuna = trial_from_model_param_setting(trial, model_param)
        if not is_valid_model(model_name, model_param_optuna):
            # このtrialは無効としてskip
            return float('inf')

        model = audio_crowd_model(model_name, model_param_optuna).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=model_param.get('lr', 1e-5),
                                     weight_decay=model_param.get('weight_decay', 0.0))
        criterion = nn.MSELoss()
        if 'criterion' in model_param.keys():
            assert model_param['criterion'] in LOSS_LIST
            criterion = nn.L1Loss() if model_param['criterion'] == 'MAE' else nn.MSELoss()

        train_loss, test_loss = [], []
        for ep in range(epoch):
            tr_loss_tmp = model_train_sw(model, train_dataloader, criterion, optimizer, ep, swc, device)
            ts_loss_tmp = model_test_sw(model, test_dataloader, criterion, ep, swc, device)
            # tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, ep, dev, verbose=True)
            # ts_loss_tmp = model_test(model, test_dataloader, criterion, ep, dev, verbose=True)
            train_loss.append(tr_loss_tmp)
            test_loss.append(ts_loss_tmp)

        each_folder = f'{model_folder}/trial{trial._trial_id}'
        if not os.path.exists(each_folder):
            os.makedirs(each_folder, exist_ok=True)
        view_loss(train_loss, test_loss, f'{each_folder}/loss.png')

        target_np, output_np = model_predict(model, train_dataloader, dev, verbose=False)
        plot_result(each_folder, target_np, output_np, col_list, label='train', log_scale=log_scale)
        write_result(each_folder, target_np, output_np, col_list, label='train', log_scale=log_scale)

        target_np, output_np = model_predict(model, test_dataloader, dev, verbose=False)
        plot_result(each_folder, target_np, output_np, col_list, label='test', log_scale=log_scale)
        write_result(each_folder, target_np, output_np, col_list, label='test', log_scale=log_scale)

        with open(f'{each_folder}/model_params.json', 'w') as fp_:
            json.dump(model_param_optuna, fp_, indent=4)

        return test_loss[-1]

    optuna_sampler = optuna.samplers.TPESampler(seed=42) if sampler == 'tpe' else optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna_sampler,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = model_param_from_best_trial(model_param_setting=model_param,
                                              best_params_optuna=study.best_params)
    with open(f'{model_folder}/best_params.json', 'w') as fp:
        json.dump(best_params, fp, indent=4)
    df = study.trials_dataframe()
    df.to_csv(f'{model_folder}/optuna_trials.csv', index=False)
    return


def copy_json(json_path, model_folder):
    json_name = os.path.basename(json_path)
    shutil.copyfile(args.input_config_json, f'{model_folder}/{json_name}')
    logger.info(f'Copied input json for model param: {model_folder}/{json_name}')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=[
        'train', 'predict', 'tuning', 'finetune',
        'train_multitask', 'predict_multitask', 'tuning_multitask', 'finetune_multitask',
        'train_multichannel', 'predict_multichannel', 'tuning_multichannel', 'finetune_multichannel'
    ],)
    parser.add_argument('-c', '--input-config-json', type=str)
    parser.add_argument('-d', '--device', type=str, default=None)
    args = parser.parse_args()

    with open(args.input_config_json, 'r') as f:
        cf = json.load(f)
    opt_ = cf.get('option', args.option)
    dev_ = cf.get('device', args.device)

    if opt_ == 'train':
        cf = cf['train']
        audio_crowd_training(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf.get('valid_folder_list', None),
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=cf.get('target', ['count']),
            epoch=cf['epoch'],
            dev=dev_,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )
        copy_json(args.input_config_json, cf['model_folder'])

    elif opt_ == 'predict':
        cf = cf['predict']
        audio_crowd_prediction(
            input_folder_list=cf['input_folder_list'],
            output_folder=cf['output_folder'],
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=cf.get('target', ['count']),
            dev=dev_,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )
        copy_json(args.input_config_json, cf['model_folder'])

    elif opt_ == 'tuning':
        cf = cf['optuna']
        audio_crowd_tuning(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf.get('valid_folder_list', None),
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=cf.get('target', ['count']),
            epoch=cf['epoch'],
            n_trials=cf.get('n_trials', 1000),
            sampler=cf.get('sampler', 'tpe'),
            dev=dev_,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )
        copy_json(args.input_config_json, cf['model_folder'])

    elif opt_ == 'finetune':
        cf = cf['finetune']
        audio_crowd_finetune(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf.get('valid_folder_list', None),
            model_folder=cf['model_folder'],
            pretrain_model_folder=cf['pretrain_model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=cf.get('target', ['count']),
            epoch=cf['epoch'],
            dev=dev_,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )
        copy_json(args.input_config_json, cf['model_folder'])

    elif opt_ == 'train_multitask':
        cf = cf['train']
        audio_crowd_training_multitask(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf.get('valid_folder_list', None),
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=cf.get('target', ['count']),
            epoch=cf['epoch'],
            weight=cf['weight'],
            tasks=cf['tasks'],
            dev=dev_,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )
        copy_json(args.input_config_json, cf['model_folder'])

    elif opt_ == 'predict_multitask':
        logger.error('multi_predict is not implemented yet.')
        pass

    elif opt_ == 'tuning_multitask':
        cf = cf['optuna']
        audio_crowd_tuning_multitask(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf.get('valid_folder_list', None),
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=cf.get('target', ['count']),
            epoch=cf['epoch'],
            n_trials=cf.get('n_trials', 1000),
            sampler=cf.get('sampler', 'tpe'),
            weight=cf['weight'],
            tasks=cf['tasks'],
            dev=dev_,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )
        copy_json(args.input_config_json, cf['model_folder'])

    elif opt_ == 'finetune_multitask':
        logger.error('multi_finetune is not implemented yet.')
        pass

    elif opt_ == 'train_multichannel':
        cf = cf['train']
        audio_crowd_training_multichannel(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf.get('valid_folder_list', None),
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            epoch=cf['epoch'],
            dev=dev_,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )

    elif opt_ == 'tuning_multichannel':
        cf = cf['optuna']
        audio_crowd_tuning_multichannel(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf.get('valid_folder_list', None),
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            epoch=cf['epoch'],
            n_trials=cf.get('n_trials', 1000),
            sampler=cf.get('sampler', 'tpe'),
            dev=dev_,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )
        copy_json(args.input_config_json, cf['model_folder'])