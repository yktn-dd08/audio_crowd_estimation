import re
import json
import glob
import argparse
import os.path
import optuna
import librosa
import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from model.cnn_model import *
from model.transformer import *
from model.vggish_model import *
from torchaudio.prototype.pipelines import VGGISH
from common.logger import get_logger
from analysis.model_common import *


FS = 16000
MODEL_LIST = ['VGGishLinear', 'VGGishTransformer', 'SimpleCNN', 'SimpleCNN2', 'AreaSpecificCNN', 'AreaSpecificCNN2',
              'ASTRegressor', 'Conv1dTransformer']
SAMPLER_LIST = ['random', 'tpe']
TASK_PATTERN = r'range_(\d+)-(\d+)'
logger = get_logger('analysis.model_training')


def read_wav(input_folder, channel_num=1):
    with open(f'{input_folder}/signal_info.json') as f:
        signal_info = json.load(f)
    assert len(signal_info['microphone']) >= channel_num, 'channel_num must be less than microphone number'
    sig_max = signal_info['signal_max']
    result = []
    for ch in range(channel_num):
        wav_path = signal_info['info'][f'ch{ch}']['wav_path']
        fs, signal = sp.io.wavfile.read(wav_path)
        signal *= sig_max
        if fs != FS:
            signal = librosa.resample(y=signal, orig_sr=fs, target_sr=FS)
        result.append(signal)
    return np.array(result)


def read_crowd(input_folder, channel_num=1):
    with open(f'{input_folder}/signal_info.json') as f:
        signal_info = json.load(f)
    assert len(signal_info['microphone']) >= channel_num, 'channel_num must be less than microphone number'
    # distance_list = signal_info['distance_list'] if 'distance_list' in signal_info.keys() else None
    result = []
    for ch in range(channel_num):
        crowd_path = signal_info['info'][f'ch{ch}']['each_crowd']
        df = pd.read_csv(crowd_path)
        result.append({col: df[col].tolist() for col in df.columns})
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


def read_logmel_torch(input_folder, target=None, tasks=None,
                      channel_num=1, time_sec=1, log_scale=True, time_agg=False):
    if target is None:
        target = ['count']
    assert channel_num == 1, 'Can input only single channel signal.'
    signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]
    crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]
    if tasks is not None:
        for task in tasks:
            crowd[task] = calc_other_task(crowd=crowd, task_name=task)
    time_length = min(int(len(signal) / FS), len(crowd[target[0]]))
    x = [trans_logmel(signal[FS * (t+1-time_sec):FS * (t + 1)], FS) for t in range(time_sec - 1,time_length)]
    x = torch.stack(x)
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


def read_logmel_torch_ast(input_folder, model_param, target=None, tasks=None,
                          channel_num=1, time_sec=1, log_scale=True, time_agg=False):
    # TODO 修正、関係する関数も全て引数から見直し
    if target is None:
        target = ['count']
    assert channel_num == 1, 'Can input only single channel signal.'
    signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]
    crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]
    if tasks is not None:
        for task in tasks:
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

def read_logmel_torch_multi_task(input_folder, target, tasks,
                                 channel_num=1, time_sec=1, log_scale=True, time_agg=False):
    assert channel_num == 1, 'Can input only single channel signal.'
    signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]
    crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]
    for task in tasks:
        crowd[task] = calc_other_task(crowd=crowd, task_name=task)
    time_length = min(int(len(signal) / FS), len(crowd[tasks[0]]))
    x = [trans_logmel(signal[FS * (t+1-time_sec):FS * (t + 1)], FS) for t in range(time_sec - 1,time_length)]
    x = torch.stack(x)
    y_np = np.array([crowd[tg][time_sec-1:time_length] for tg in tasks]).T
    task_np = np.array([crowd[ts][time_sec-1:time_length] for ts in tasks]).T
    if time_agg:
        y_np = np.array([[sum(crowd[tg][t + 1 - time_sec:t + 1]) / time_sec for t in range(time_sec - 1, time_length)]
                         for tg in target]).T
        task_np = np.array([[sum(crowd[ts][t + 1 - time_sec:t + 1]) / time_sec
                             for t in range(time_sec - 1, time_length)]
                            for ts in tasks]).T
    if log_scale:
        y_np = np.log(y_np + 1.0)
        task_np = np.log(task_np + 1.0)
    y, task = torch.Tensor(y_np), torch.Tensor(task_np)
    return x, y, task


def read_logmel_torch_vgg(input_folder, target=None, tasks=None,
                          channel_num=1, time_sec=1, log_scale=True, time_agg=False):
    if target is None:
        target = ['count']
    assert channel_num == 1, 'Can input only single channel signal.'
    signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]
    crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]
    if tasks is not None:
        for task in tasks:
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


def load_dataset(input_folder_list, valid_folder_list, model_name, model_param, time_agg, target,
                 valid_flag=True, tasks=None):
    x, y, task = None, None, None
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
                model_param=model_param
            )
            x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
            y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)

        else:
            tmp_x, tmp_y, tmp_task = read_logmel(
                model_name=model_name,
                input_folder=input_folder,
                target=target,
                channel_num=1,
                tasks=tasks,
                time_sec=model_param['time_sec'],
                time_agg=time_agg
            )
            x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
            y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)
            task = tmp_task if i == 0 else torch.cat([task, tmp_task], dim=0)

    # 検証用のデータがいらない場合 (予測のみ行う場合)
    if not valid_flag:
        if tasks is None:
            return x, y
        else:
            return x, y, task

    valid_x, valid_y, valid_task = None, None, None

    if valid_folder_list is None:
        # valid_folder_listがNoneの時はinput_folder_listのデータをランダムで学習用、評価用に分割する
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
                valid_x = tmp_x if i == 0 else torch.cat([valid_x, tmp_x], dim=0)
                valid_y = tmp_y if i == 0 else torch.cat([valid_y, tmp_y], dim=0)

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
                valid_x = tmp_x if i == 0 else torch.cat([valid_x, tmp_x], dim=0)
                valid_y = tmp_y if i == 0 else torch.cat([valid_y, tmp_y], dim=0)
                valid_task = tmp_task if i == 0 else torch.cat([valid_task, tmp_task], dim=0)

    if tasks is None:
        return (x, y), (valid_x, valid_y)
    else:
        return (x, y, task), (valid_x, valid_y, valid_task)


def audio_crowd_model(model_name, model_param):
    if model_name == 'VGGishLinear':
        return VGGishLinear2(frame_num=model_param['time_sec'],
                             out_features=model_param['out_features'],
                             pre_trained=model_param['pre_trained'])
    elif model_name == 'VGGishTransformer':
        return VGGishTransformer(frame_num=model_param['time_sec'],
                                 token_dim=model_param['token_dim'],
                                 n_head=model_param['n_head'],
                                 h_dim=model_param['h_dim'],
                                 layer_num=model_param['layer_num'],
                                 out_features=model_param['out_features'],
                                 pre_trained=model_param['pre_trained'])
    elif model_name == 'SimpleCNN':
        return SimpleCNN(frame_num=model_param['frame_num'],
                         freq_num=model_param['freq_num'],
                         kernel_size=model_param['kernel_size'])
    elif model_name == 'SimpleCNN2':
        return SimpleCNN2(frame_num=model_param['frame_num'],
                          freq_num=model_param['freq_num'],
                          kernel_size=model_param['kernel_size'],
                          dilation_size=model_param['dilation_size'],
                          layer_num=model_param['layer_num'],
                          inter_ch=model_param['inter_ch'])
    elif model_name == 'AreaSpecificCNN':
        return AreaSpecificCNN(task_num=model_param['task_num'],
                               freq_num=model_param['freq_num'],
                               frame_num=model_param['frame_num'],
                               kernel_size=model_param['kernel_size'],
                               dilation_size=model_param['dilation_size'],
                               layer_num=model_param['layer_num'],
                               inter_ch=model_param['inter_ch'],
                               pool_size=model_param['pool_size'])
    elif model_name == 'AreaSpecificCNN2':
        return AreaSpecificCNN2(task_num=model_param['task_num'],
                                freq_num=model_param['freq_num'],
                                frame_num=model_param['frame_num'],
                                common_kernel_size=model_param['common_kernel_size'],
                                kernel_size=model_param['kernel_size'],
                                common_dilation_size=model_param['common_dilation_size'],
                                dilation_size=model_param['dilation_size'],
                                common_layer_num=model_param['common_layer_num'],
                                layer_num=model_param['layer_num'],
                                common_inter_ch=model_param['common_inter_ch'],
                                inter_ch=model_param['inter_ch'],
                                common_pool_size=model_param['common_pool_size'],
                                pool_size=model_param['pool_size'])
    elif model_name == 'ASTRegressor':
        return ASTRegressor(feat_num=model_param['feat_num'],
                            drop_out=model_param['drop_out'],
                            model_name=model_param['model_name'],
                            finetune=model_param['finetune'])
    elif model_name == 'Conv1dTransformer':
        return Conv1dTransformer(freq_num=model_param['freq_num'],
                                 frame_num=model_param['frame_num'],
                                 kernel_size=model_param['kernel_size'],
                                 dilation_size=model_param['dilation_size'],
                                 pool_size=model_param['pool_size'],
                                 token_dim=model_param['token_dim'],
                                 n_head=model_param['n_head'],
                                 drop_out=model_param['drop_out'],
                                 layer_num=model_param['layer_num'],
                                 pe_flag=model_param['pe_flag'],
                                 feat_num=model_param['feat_num'])
    else:
        Exception(f'Model: {model_name} is not implemented.')


def trial_from_model_param_setting(trial: optuna.Trial, model_param_setting: dict):
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
                elif 'float' in val['size']:
                    model_param_optuna[name] = [
                        [
                            trial.suggest_float(name=f'{name}_{i0}_{i1}',
                                                low=val['low'],
                                                high=val['high'])
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
                else:
                    Exception(f'Invalid model parameter setting, {name}: {val}')

            elif val['type'].count('list') == 0:
                if val['type'] == 'int':
                    model_param_optuna[name] = trial.suggest_int(name=name, low=val['low'], high=val['high'])
                elif val['type'] == 'float':
                    model_param_optuna[name] = trial.suggest_float(name=name, low=val['low'], high=val['high'])
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
    model_param_setting optunaに設定する際に入力したJSONデータ
    best_params_optuna optuna探索により得られた最良の結果

    Returns 探索結果のモデルパラメタを格納したJSONデータ
    -------

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
    model_name モデル名：現状SimpleCNN2, MultiTaskCNNのみ検証の必要がある
    model_param モデルパラメタ

    Returns モデルがvalidかどうか
    -------

    """
    if model_name == 'SimpleCNN2':
        return SimpleCNN2.is_valid(
            frame_num=model_param['frame_num'],
            freq_num=model_param['freq_num'],
            kernel_size=model_param['kernel_size'],
            dilation_size=model_param['dilation_size'],
            layer_num=model_param['layer_num'],
            inter_ch=model_param['inter_ch']
        )
    elif model_name == 'AreaSpecificCNN':
        return AreaSpecificCNN.is_valid(
            task_num=model_param['task_num'],
            freq_num=model_param['task_num'],
            frame_num=model_param['frame_num'],
            kernel_size=model_param['kernel_size'],
            dilation_size=model_param['dilation_size'],
            layer_num=model_param['layer_num'],
            inter_ch=model_param['inter_ch'],
            pool_size=model_param['pool_size']
        )
    elif model_name == 'AreaSpecificCNN2':
        return AreaSpecificCNN2.is_valid(
            task_num=model_param['task_num'],
            freq_num=model_param['freq_num'],
            frame_num=model_param['frame_num'],
            common_kernel_size=model_param['common_kernel_size'],
            kernel_size=model_param['kernel_size'],
            common_dilation_size=model_param['common_dilation_size'],
            dilation_size=model_param['dilation_size'],
            common_layer_num=model_param['common_layer_num'],
            layer_num=model_param['layer_num'],
            common_inter_ch=model_param['common_inter_ch'],
            inter_ch=model_param['inter_ch'],
            common_pool_size=model_param['common_pool_size'],
            pool_size=model_param['pool_size']
        )
    return True


def write_result(folder, target, output, label):
    target_df = pd.DataFrame(target)
    target_df.columns = [f'target{i}' for i in range(len(target_df.columns))]
    output_df = pd.DataFrame(output)
    output_df.columns = [f'predict{i}' for i in range(len(output_df.columns))]
    res_df = pd.concat([target_df, output_df], axis=1)
    res_df.to_csv(f'{folder}/result_{label}.csv', index=False)
    calculate_accuracy(target, output, f'{folder}/acc_{label}.json')
    return


def audio_crowd_training(input_folder_list, valid_folder_list,
                         model_folder, model_name, model_param, target, epoch, dev=None,
                         log_scale=True, time_agg=False, batch_size=64):
    device = get_device(dev)
    logger.info(f'Start {model_name} Training: {device} - Columns: {target}')
    (x, y), (valid_x, valid_y) = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        target=target
    )
    # x, y = None, None
    # for i, input_folder in enumerate(input_folder_list):
    #     logger.info(f'Reading Training Folders: {input_folder}')
    #     tmp_x, tmp_y = read_logmel(model_name=model_name, input_folder=input_folder, target=target, channel_num=1,
    #                                time_sec=model_param['time_sec'], time_agg=time_agg)
    #     x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
    #     y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)
    #
    # valid_x, valid_y = None, None
    # if valid_folder_list is None:
    #     tr_idx, ts_idx = train_test_split(range(len(y)), test_size=0.2, random_state=0)
    #     valid_x, valid_y = x[ts_idx], y[ts_idx]
    #     x, y = x[tr_idx], y[tr_idx]
    # else:
    #     for i, valid_folder in enumerate(valid_folder_list):
    #         logger.info(f'Reading Valid Folders: {valid_folder}')
    #         tmp_x, tmp_y = read_logmel(model_name=model_name, input_folder=valid_folder, target=target, channel_num=1,
    #                                    time_sec=model_param['time_sec'], time_agg=time_agg)
    #         valid_x = tmp_x if i == 0 else torch.cat([valid_x, tmp_x], dim=0)
    #         valid_y = tmp_y if i == 0 else torch.cat([valid_y, tmp_y], dim=0)

    train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
    test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    if 'out_features' not in model_param:
        model_param['out_features'] = len(target)
    model = audio_crowd_model(model_name, model_param).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=2.7e-09)
    criterion = nn.MSELoss()

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
    if log_scale:
        scatter_plot(target_np, output_np, f'{model_folder}/train_scatter_log.png')
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
    scatter_plot(target_np, output_np, f'{model_folder}/train_scatter.png')

    target_np, output_np = model_predict(model, test_dataloader)
    if log_scale:
        scatter_plot(target_np, output_np, f'{model_folder}/scatter_log.png')
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
    scatter_plot(target_np, output_np, f'{model_folder}/scatter.png')
    write_result(model_folder, target_np, output_np, label=target[0])
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
    (x, y), (valid_x, valid_y) = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        target=target
    )
    # x, y = None, None
    # for i, input_folder in enumerate(input_folder_list):
    #     logger.info(f'Reading Training Folders: {input_folder}')
    #     tmp_x, tmp_y = read_logmel(model_name=model_name, input_folder=input_folder, target=target, channel_num=1,
    #                                time_sec=model_param['time_sec'], time_agg=time_agg)
    #     x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
    #     y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)
    #
    # valid_x, valid_y = None, None
    # if valid_folder_list is None:
    #     tr_idx, ts_idx = train_test_split(range(len(y)), test_size=0.2, random_state=0)
    #     valid_x, valid_y = x[ts_idx], y[ts_idx]
    #     x, y = x[tr_idx], y[tr_idx]
    # else:
    #     for i, valid_folder in enumerate(valid_folder_list):
    #         logger.info(f'Reading Valid Folders: {valid_folder}')
    #         tmp_x, tmp_y = read_logmel(model_name=model_name, input_folder=valid_folder, target=target, channel_num=1,
    #                                    time_sec=model_param['time_sec'], time_agg=time_agg)
    #         valid_x = tmp_x if i == 0 else torch.cat([valid_x, tmp_x], dim=0)
    #         valid_y = tmp_y if i == 0 else torch.cat([valid_y, tmp_y], dim=0)

    train_dataset = torch.utils.data.TensorDataset(x.to(device), y.to(device))
    test_dataset = torch.utils.data.TensorDataset(valid_x.to(device), valid_y.to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    if 'out_features' not in model_param:
        model_param['out_features'] = len(target)
    # model = audio_crowd_model(model_name, model_param).to(device)

    def objective(trial: optuna.Trial):
        model_param_optuna = trial_from_model_param_setting(trial, model_param)
        # model_param_optuna = {}
        # for name, val in model_param.items():
        #     if not isinstance(val, dict):
        #         model_param_optuna[name] = val
        #     else:
        #         if 'list' in val['type']:
        #             list_size = val['size'] if isinstance(val['size'], int) else model_param_optuna[val['size']]
        #             if 'int' in val['type']:
        #                 model_param_optuna[name] = [trial.suggest_int(name=f'{name}_{idx}',
        #                                                               low=val['low'],
        #                                                               high=val['high'])
        #                                             for idx in range(list_size)]
        #             elif 'float' in val['type']:
        #                 model_param_optuna[name] = [trial.suggest_float(name=f'{name}_{idx}',
        #                                                                 low=val['low'],
        #                                                                 high=val['high'])
        #                                             for idx in range(list_size)]
        #             else:
        #                 Exception(f'Invalid model parameter setting, {name}: {val}')
        #         else:
        #             if val['type'] == 'int':
        #                 model_param_optuna[name] = trial.suggest_int(name=name, low=val['low'], high=val['high'])
        #             elif val['type'] == 'float':
        #                 model_param_optuna[name] = trial.suggest_float(name=name, low=val['low'], high=val['high'])
        #             else:
        #                 Exception(f'Invalid model parameter setting, {name}: {val}')
        #
        if not is_valid_model(model_name, model_param_optuna):
            # このtrialは無効としてskip
            return float('inf')
        # if model_name == 'SimpleCNN2':
        #     if not SimpleCNN2.is_valid(
        #             frame_num=model_param_optuna['frame_num'],
        #             freq_num=model_param_optuna['freq_num'],
        #             kernel_size=model_param_optuna['kernel_size'],
        #             dilation_size=model_param_optuna['dilation_size'],
        #             layer_num=model_param_optuna['layer_num'],
        #             inter_ch=model_param_optuna['inter_ch']
        #     ):
        #         # このtrialは無効としてskip
        #         return float('inf')

        model = audio_crowd_model(model_name, model_param_optuna).to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=2.7e-09)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-05)
        criterion = nn.MSELoss()

        train_loss, test_loss = [], []
        for ep in range(epoch):
            tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, ep, verbose=False)
            ts_loss_tmp = model_test(model, test_dataloader, criterion, ep, verbose=False)
            train_loss.append(tr_loss_tmp)
            test_loss.append(ts_loss_tmp)

        each_folder = f'{model_folder}/trial{trial._trial_id}'
        if not os.path.exists(each_folder):
            os.makedirs(each_folder, exist_ok=True)
        view_loss(train_loss, test_loss, f'{each_folder}/loss.png')

        target_np, output_np = model_predict(model, train_dataloader, verbose=False)
        if log_scale:
            scatter_plot(target_np, output_np, f'{each_folder}/train_scatter_log.png')
            target_np = np.exp(target_np) - 1
            output_np = np.exp(output_np) - 1
        scatter_plot(target_np, output_np, f'{each_folder}/train_scatter.png')

        target_np, output_np = model_predict(model, test_dataloader, verbose=False)
        if log_scale:
            scatter_plot(target_np, output_np, f'{each_folder}/scatter_log.png')
            target_np = np.exp(target_np) - 1
            output_np = np.exp(output_np) - 1
        scatter_plot(target_np, output_np, f'{each_folder}/scatter.png')

        tried_params = model_param_from_best_trial(model_param, model_param_optuna)
        with open(f'{each_folder}/model_params.json', 'w') as fp_:
            json.dump(tried_params, fp_)
        return test_loss[-1]
    optuna_sampler = optuna.samplers.TPESampler(seed=42) if sampler == 'tpe' else optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna_sampler,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    # best_params_optuna = study.best_params
    # best_params = {}
    best_params = model_param_from_best_trial(model_param_setting=model_param,
                                              best_params_optuna=study.best_params)
    # for name, val in model_param.items():
    #     if not isinstance(val, dict):
    #         best_params[name] = val
    #     else:
    #         if 'list' in val['type']:
    #             if isinstance(val['size'], str):
    #                 if isinstance(model_param[val['size']], int):
    #                     # 要素数が固定値でなく、別パラメータを参照しており、またoptunaで探索させていない場合
    #                     list_size = model_param[val['size']]
    #                 else:
    #                     # optunaで要素数を探索させている場合はoptunaの結果から取得
    #                     list_size = best_params_optuna[val['size']]
    #             else:
    #                 # 要素数が固定値の場合
    #                 list_size = val['size']
    #             best_params[name] = [best_params_optuna[f'{name}_{idx}'] for idx in range(list_size)]
    #         else:
    #             best_params[name] = best_params_optuna[name]
    with open(f'{model_folder}/best_params.json', 'w') as fp:
        json.dump(best_params, fp)
    df = study.trials_dataframe()
    df.to_csv(f'{model_folder}/optuna_trials.csv', index=False)
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
        target=target,
        tasks=tasks
    )
    # x, y, task = None, None, None
    # for i, input_folder in enumerate(input_folder_list):
    #     logger.info(f'Reading Training Folders: {input_folder}')
    #     tmp_x, tmp_y, tmp_task = read_logmel(model_name=model_name, input_folder=input_folder, target=target,
    #                                          tasks=tasks, channel_num=1, time_sec=model_param['time_sec'],
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
    #                                              tasks=tasks, channel_num=1, time_sec=model_param['time_sec'],
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

    def objective(trial: optuna.Trial):
        model_param_optuna = trial_from_model_param_setting(trial, model_param)
        # model_param_optuna = {}
        # for name, val in model_param.items():
        #     if not isinstance(val, dict):
        #         model_param_optuna[name] = val
        #     else:
        #         # TODO list of listの場合のプログラム、JSON仕様
        #         if 'list' in val['type']:
        #             list_size = val['size'] if isinstance(val['size'], int) else model_param_optuna[val['size']]
        #             if 'int' in val['type']:
        #                 model_param_optuna[name] = [trial.suggest_int(name=f'{name}_{idx}',
        #                                                               low=val['low'],
        #                                                               high=val['high'])
        #                                             for idx in range(list_size)]
        #             elif 'float' in val['type']:
        #                 model_param_optuna[name] = [trial.suggest_float(name=f'{name}_{idx}',
        #                                                                 low=val['low'],
        #                                                                 high=val['high'])
        #                                             for idx in range(list_size)]
        #             else:
        #                 Exception(f'Invalid model parameter setting, {name}: {val}')
        #         else:
        #             if val['type'] == 'int':
        #                 model_param_optuna[name] = trial.suggest_int(name=name, low=val['low'], high=val['high'])
        #             elif val['type'] == 'float':
        #                 model_param_optuna[name] = trial.suggest_float(name=name, low=val['low'], high=val['high'])
        #             else:
        #                 Exception(f'Invalid model parameter setting, {name}: {val}')

        if not is_valid_model(model_name, model_param_optuna):
            return float('inf')
        # if model_name == 'SimpleCNN2':
        #     if not SimpleCNN2.is_valid(frame_num=model_param_optuna['frame_num'],
        #                                freq_num=model_param_optuna['freq_num'],
        #                                kernel_size=model_param_optuna['kernel_size'],
        #                                dilation_size=model_param_optuna['dilation_size'],
        #                                layer_num=model_param_optuna['layer_num'],
        #                                inter_ch=model_param_optuna['inter_ch']):
        #         # このtrialは無効としてskip
        #         return float('inf')
        # elif model_name == 'MultiTaskCNN':
        #     if not MultiTaskCNN.is_valid(task_num=model_param_optuna['task_num'],
        #                                  frame_num=model_param_optuna['frame_num'],
        #                                  freq_num=model_param_optuna['freq_num'],
        #                                  kernel_size=model_param_optuna['kernel_size'],
        #                                  dilation_size=model_param_optuna['dilation_size'],
        #                                  layer_num=model_param_optuna['layer_num'],
        #                                  inter_ch=model_param_optuna['inter_ch'],
        #                                  pool_size=model_param_optuna['pool_size']):
        #         return float('inf')

        model = audio_crowd_model(model_name, model_param_optuna).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=2.7e-09)
        criterion = nn.MSELoss()
        task_criterion = nn.MSELoss()

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

        tried_params = model_param_from_best_trial(model_param, model_param_optuna)
        with open(f'{each_folder}/model_params.json', 'w') as fp_:
            json.dump(tried_params, fp_)
        return test_loss[-1]

    optuna_sampler = optuna.samplers.TPESampler(seed=42) if sampler == 'tpe' else optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna_sampler,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = model_param_from_best_trial(model_param, study.best_params)
    # best_params_optuna = study.best_params
    # best_params = {}
    # for name, val in model_param.items():
    #     if not isinstance(val, dict):
    #         best_params[name] = val
    #     else:
    #         if 'list' in val['type']:
    #             if isinstance(val['size'], str):
    #                 if isinstance(model_param[val['size']], int):
    #                     # 要素数が固定値でなく、別パラメータを参照しており、またoptunaで探索させていない場合
    #                     list_size = model_param[val['size']]
    #                 else:
    #                     # optunaで要素数を探索させている場合はoptunaの結果から取得
    #                     list_size = best_params_optuna[val['size']]
    #             else:
    #                 # 要素数が固定値の場合
    #                 list_size = val['size']
    #             best_params[name] = [best_params_optuna[f'{name}_{idx}'] for idx in range(list_size)]
    #         else:
    #             best_params[name] = best_params_optuna[name]
    with open(f'{model_folder}/best_params.json', 'w') as fp:
        json.dump(best_params, fp)
    df = study.trials_dataframe()
    df.to_csv(f'{model_folder}/optuna_trials.csv', index=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str,
                        choices=['train', 'predict', 'tuning', 'multi_train', 'multi_predict', 'multi_tuning'])
    parser.add_argument('-c', '--input-config-json', type=str)
    parser.add_argument('-d', '--device', type=str, default=None)
    args = parser.parse_args()

    with open(args.input_config_json, 'r') as f:
        cf = json.load(f)

    if args.option == 'train':
        cf = cf['train']
        assert cf['model_name'] in MODEL_LIST
        audio_crowd_training(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf['valid_folder_list'] if 'valid_folder_list' in cf.keys() else None,
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=['count'] if 'target' not in cf.keys() else cf['target'],
            epoch=cf['epoch'],
            dev=args.device,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )

    elif args.option == 'predict':
        cf = cf['predict']
        assert cf['model_name'] in MODEL_LIST
        audio_crowd_prediction(
            input_folder_list=cf['input_folder_list'],
            output_folder=cf['output_folder'],
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=['count'] if 'target' not in cf.keys() else cf['target'],
            dev=args.device,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )

    elif args.option == 'tuning':
        cf = cf['optuna']
        assert cf['model_name'] in MODEL_LIST
        audio_crowd_tuning(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf['valid_folder_list'] if 'valid_folder_list' in cf.keys() else None,
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=['count'] if 'target' not in cf.keys() else cf['target'],
            epoch=cf['epoch'],
            n_trials=cf['n_trials'] if 'n_trials' in cf.keys() else 1000,
            sampler=cf['sampler'] if 'sampler' in cf.keys() else 'tpe',
            dev=args.device,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )

    elif args.option == 'multi_train':
        cf = cf['train']
        assert cf['model_name'] in MODEL_LIST
        audio_crowd_training_multitask(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf['valid_folder_list'] if 'valid_folder_list' in cf.keys() else None,
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=['count'] if 'target' not in cf.keys() else cf['target'],
            epoch=cf['epoch'],
            weight=cf['weight'],
            tasks=cf['tasks'],
            dev=args.device,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )

    elif args.option == 'multi_predict':
        pass

    elif args.option == 'multi_tuning':
        cf = cf['optuna']
        assert cf['model_name'] in MODEL_LIST
        audio_crowd_tuning_multitask(
            input_folder_list=cf['input_folder_list'],
            valid_folder_list=cf['valid_folder_list'] if 'valid_folder_list' in cf.keys() else None,
            model_folder=cf['model_folder'],
            model_name=cf['model_name'],
            model_param=cf['model_param'],
            target=['count'] if 'target' not in cf.keys() else cf['target'],
            epoch=cf['epoch'],
            n_trials=cf['n_trials'] if 'n_trials' in cf.keys() else 1000,
            sampler=cf['sampler'] if 'sampler' in cf.keys() else 'tpe',
            weight=cf['weight'],
            tasks=cf['tasks'],
            dev=args.device,
            batch_size=cf['batch_size'],
            log_scale=cf['log_scale'],
            time_agg=cf['time_agg']
        )
        pass
