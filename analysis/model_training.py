import json
import glob
import argparse
import os.path

import librosa
import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from model.vggish_model import VGGishLinear
from torchaudio.prototype.pipelines import VGGISH
from common.logger import get_logger
from analysis.model_common import *


FS = 16000
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


def read_logmel_torch(input_folder, target=None, channel_num=1, time_sec=1, log_scale=True):
    if target is None:
        target = ['count']
    assert channel_num == 1, 'Can input only single channel signal.'
    signal = read_wav(input_folder=input_folder, channel_num=channel_num)[0]
    crowd = read_crowd(input_folder=input_folder, channel_num=channel_num)[0]
    logmel_func = VGGISH.get_input_processor()
    time_length = min(int(len(signal) / FS), len(crowd[target[0]]))
    # TODO time_sec > 1の時の対応
    x = [logmel_func(torch.Tensor(signal[FS * (t+1-time_sec):FS * (t + 1)])) for t in range(time_length)]
    x = torch.cat(x, dim=0)
    y_np = np.array([crowd[tg][:time_length] for tg in target]).T
    if log_scale:
        y_np = np.log(y_np + 1.0)
    y = torch.Tensor(y_np)
    return x, y


def vggish_training(input_folder_list, model_folder, target, epoch, log_scale=True, vgg_frame=1, pre_trained=True,
                    batch_size=64):
    """

    Parameters
    ----------
    input_folder_list
    model_folder
    target
    epoch
    log_scale
    vgg_frame
    pre_trained
    batch_size

    Returns
    -------

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if target is None:
        target = ['count']
    logger.info(f'Start VGGishLinear Training: {device} - Columns: {target}')
    x, y = None, None
    for i, input_folder in enumerate(input_folder_list):
        logger.info(f'Reading Folders: {input_folder}')
        tmp_x, tmp_y = read_logmel_torch(input_folder=input_folder, target=target, channel_num=1)
        x = tmp_x if i == 0 else torch.cat([x, tmp_x], dim=0)
        y = tmp_y if i == 0 else torch.cat([y, tmp_y], dim=0)

    model = VGGishLinear(vgg_frame=vgg_frame, out_features=len(target), pre_trained=pre_trained).to(device)
    tr_idx, ts_idx = train_test_split(range(len(y)), test_size=0.2, random_state=0)
    train_dataset = torch.utils.data.TensorDataset(x[tr_idx].to(device), y[tr_idx].to(device))
    test_dataset = torch.utils.data.TensorDataset(x[ts_idx].to(device), y[ts_idx].to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # TODO optunaでチューニング
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
    torch.save(model.state_dict(), f'{model_folder}/vggish_model.pt')

    target_np, output_np = model_predict(model, train_dataloader)
    scatter_plot(target_np, output_np, f'{model_folder}/train_scatter_log.png')
    if log_scale:
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
    scatter_plot(target_np, output_np, f'{model_folder}/train_scatter.png')

    target_np, output_np = model_predict(model, test_dataloader)
    scatter_plot(target_np, output_np, f'{model_folder}/scatter_log.png')
    if log_scale:
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
    scatter_plot(target_np, output_np, f'{model_folder}/scatter.png')
    return


def vggish_prediction(input_folder_list, output_folder):
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['train', 'predict'])
    parser.add_argument('-c', '--input-config-json', type=str)
    args = parser.parse_args()

    with open(args.input_config_json, 'r') as f:
        cf = json.load(f)
    if args.option == 'train':
        file_setting = cf['file']
        param_setting = cf['train']
        vggish_training(input_folder_list=file_setting['input_folder_list'],
                        model_folder=file_setting['model_folder'],
                        target=param_setting['target'],
                        epoch=param_setting['epoch'],
                        log_scale=param_setting['log_scale'],
                        vgg_frame=param_setting['vgg_frame'],
                        pre_trained=param_setting['pre_trained'],
                        batch_size=param_setting['batch_size'])
    elif args.option == 'predict':
        pass
    pass
