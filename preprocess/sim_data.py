import glob
import json
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import py2shpss as hpss
from tqdm import tqdm
from scipy.io import wavfile
from joblib import Parallel, delayed
from librosa.feature import melspectrogram

MEL_EPS = 1.0e-35


def hpss_feature(signal, fs, feat='ohvp'):
    hpss2s = hpss.twostageHPSS(samprate=fs)
    ll = len(signal)
    ms = np.max(signal)
    h, v, p = hpss2s(signal/ms)
    h = h[0:ll] * ms if len(h) > ll else np.concatenate([h * ms, np.zeros(ll - len(h))])
    v = v[0:ll] * ms if len(v) > ll else np.concatenate([v * ms, np.zeros(ll - len(v))])
    p = p[0:ll] * ms if len(p) > ll else np.concatenate([p * ms, np.zeros(ll - len(p))])
    res = []
    for f in feat:
        if f == 'o':
            res.append(f)
        elif f == 'h':
            res.append(h)
        elif f == 'v':
            res.append(v)
        elif f == 'p':
            res.append(p)
        else:
            raise Exception('feat must include "ohvp"')
    return res


def each_feat_calc_old(crowd, signal, fs, index):
    y = crowd['crowd'].loc[index]
    tmp = signal[int(fs*index):int(fs*(index+1))]
    s = np.zeros(fs)
    s[0:len(tmp)] = tmp
    hpss2s = hpss.twostageHPSS(samprate=fs)
    ll = len(s)
    ms = np.max(s)
    h, v, p = hpss2s(s/ms)
    h = h[0:ll] * ms if len(h) > ll else np.concatenate([h * ms, np.zeros(ll - len(h))])
    v = v[0:ll] * ms if len(v) > ll else np.concatenate([v * ms, np.zeros(ll - len(v))])
    p = p[0:ll] * ms if len(p) > ll else np.concatenate([p * ms, np.zeros(ll - len(p))])
    feat = np.stack([np.log10(melspectrogram(y=d, sr=fs)+MEL_EPS) for d in [s, h, v, p]])
    return y, feat


def calculate_feature_old(input_folder, output_folder):
    crowd = pd.read_csv(f'{input_folder}/crowd.csv')
    wav_list = glob.glob(f'{input_folder}/sim*.wav')
    for i, w in enumerate(wav_list):
        fs, signal = wavfile.read(w)
        # signal += (np.random.random(len(signal)) * 2.0 - 1.0) * 1.0e-35
        xy = [each_feat_calc_old(crowd, signal, fs, t) for t in tqdm(range(len(crowd)))]
        y = np.array([f[0] for f in xy])
        x = np.stack([f[1] for f in xy])
        feature = {'y': y, 'x': x}
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(f'{output_folder}/feature{i}.pickle', 'wb') as f:
            pickle.dump(feature, f)
    return


def calculate_crowd_with_duration(crowd_df, start_time, end_time):
    col_list = [col for col in crowd_df.columns if col != 't' and col != 'time_index']
    time = 0.0
    crowd = {col: 0.0 for col in col_list}

    ext_df = crowd_df[(start_time <= crowd_df['t']) & (crowd_df['t'] < end_time)].sort_values('t')
    ext_before = crowd_df[crowd_df['t'] < start_time]
    ext_after = crowd_df[crowd_df['t'] >= end_time]

    # in case of no record filtered
    if len(ext_df) == 0:
        tmp = crowd_df.loc[[ext_before.idxmax()['t']]]
        tmp['t'] = start_time
        ext_df = pd.concat([ext_df, tmp], axis=0)

    # counting crowd at start_time
    if len(ext_before) > 0 and ext_df['t'].min() > start_time:
        tmp = crowd_df.loc[[ext_before.idxmax()['t']]]
        tmp['t'] = start_time
        ext_df = pd.concat([ext_df, tmp], axis=0)
    elif len(ext_before) > 0 and ext_df['t'].min() > start_time:
        tmp = pd.DataFrame([{col: 0.0 for col in ext_df.columns}])
        tmp['t'] = start_time
        ext_df = pd.concat([ext_df, tmp], axis=0)

    # counting crowd at end_time
    if len(ext_after) > 0:
        tmp = crowd_df.loc[[ext_after.idxmin()['t']]]
        tmp['t'] = end_time
        ext_df = pd.concat([ext_df, tmp], axis=0)
    else:
        tmp = pd.DataFrame([{col: 0.0 for col in ext_df.columns}])
        tmp['t'] = end_time
        ext_df = pd.concat([ext_df, tmp], axis=0)
    ext_df = ext_df.sort_values('t').reset_index(drop=True)

    # calculating weighted crowd number
    for i in range(len(ext_df) - 1):
        delta_t = ext_df.iloc[i + 1]['t'] - ext_df.iloc[i]['t']
        time += delta_t
        for col in col_list:
            crowd[col] += delta_t * ext_df.iloc[i][col]
    return {k: v / time for k, v in crowd.items()}


def calculate_log_mel_feature(input_folder, output_folder, duration=1.0, step=1.0, feat=None):
    wav_list = glob.glob(f'{input_folder}/sim_mic*.wav')
    csv_list = glob.glob(f'{input_folder}/crowd_mic*.csv')
    sorted(wav_list)
    sorted(csv_list)
    assert len(wav_list) == len(csv_list), f'File Error: # of wav is {len(wav_list)}, # of csv is {len(csv_list)}'

    fs, _ = wavfile.read(wav_list[0])

    frame_shift = int(step * fs)
    frame_num = int(duration * fs)
    result = []

    with open(f'{input_folder}/signal_info.json', 'r') as f:
        signal_info = json.load(f)

    for i, (wav_path, csv_path) in enumerate(zip(wav_list, csv_list)):
        fs, signal = wavfile.read(wav_path)
        crowd_mic = pd.read_csv(csv_path)
        signal *= signal_info['signal_max']

        feature_array = []
        crowd_array = []
        for t_idx in tqdm(np.arange(0, len(signal), frame_shift), desc=f'[microphone {i}/{len(wav_list)}]'):
            signal_sub = signal[t_idx:t_idx+frame_num]
            if feat is None:
                log_mel = np.log10(melspectrogram(y=signal_sub, sr=fs) + MEL_EPS)[np.newaxis]
                feature_array.append(log_mel)
            else:
                feature = hpss_feature(signal_sub, fs, feat)
                log_mel = np.stack([np.log10(melspectrogram(y=d, sr=fs) + MEL_EPS) for d in feature])
                feature_array.append(log_mel)

            crowd_dict = calculate_crowd_with_duration(crowd_mic,
                                                       start_time=t_idx / fs,
                                                       end_time=(t_idx + frame_num) / fs)
            crowd_array.append(crowd_dict)
        result.append({'microphone': i, 'crowd': crowd_array, 'feature': feature_array,
                       'csv_path': csv_path, 'wav_path': wav_path})
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(f'{output_folder}/feature.pickle', 'wb') as f:
        pickle.dump(result, f)
    return


def calculate_feature_mp(input_folder, output_folder):
    crowd = pd.read_csv(f'{input_folder}/crowd.csv')
    wav_list = glob.glob(f'{input_folder}/sim*.wav')
    for i, w in enumerate(wav_list):
        fs, signal = wavfile.read(w)

        def _each_feat_calc(index):
            y = crowd['crowd'].loc[index]
            tmp = signal[int(fs * index):int(fs * (index + 1))]
            s = np.zeros(fs)
            s[0:len(tmp)] = tmp
            hpss2s = hpss.twostageHPSS(samprate=fs)
            ll = len(s)
            ms = np.max(s)
            h, v, p = hpss2s(s / ms)
            h = h[0:ll] * ms if len(h) > ll else np.concatenate([h * ms, np.zeros(ll - len(h))])
            v = v[0:ll] * ms if len(v) > ll else np.concatenate([v * ms, np.zeros(ll - len(v))])
            p = p[0:ll] * ms if len(p) > ll else np.concatenate([p * ms, np.zeros(ll - len(p))])
            feat = np.stack([np.log10(melspectrogram(y=d, sr=fs) + 1.0e-35) for d in [s, h, v, p]])
            return (y, feat), index

        # xy = [each_feat_calc(crowd, signal, fs, t) for t in tqdm(range(len(crowd)))]
        res_list = Parallel(n_jobs=-1)(delayed(_each_feat_calc)(index) for index in tqdm(range(len(crowd))))
        sorted(res_list, key=lambda t: t[1])
        xy = [rl[0] for rl in res_list]
        y = np.array([f[0] for f in xy])
        x = np.stack([f[1] for f in xy])
        feature = {'y': y, 'x': x}
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(f'{output_folder}/feature{i}.pickle', 'wb') as f:
            pickle.dump(feature, f)
    return


def sim_plot(input_folder, output_folder, time_unit=10):
    crowd = pd.read_csv(f'{input_folder}/crowd.csv')
    fs, signal = wavfile.read(f'{input_folder}/sim0.wav')
    iteration = int(crowd['t'].max() / time_unit) + 1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(iteration):
        cr = crowd[(crowd['t'] >= i * time_unit) & (crowd['t'] < (i + 1) * time_unit)]
        s = signal[i*time_unit*fs:(i+1)*time_unit*fs]

        log_mel = np.log10(melspectrogram(y=s, sr=fs))
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.imshow(log_mel, cmap='rainbow')
        ax1.invert_yaxis()
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(cr['crowd'].values)
        plt.savefig(f'{output_folder}/logmel_crowd{i:04}.png')
        plt.close()

    log_mel = np.log10(melspectrogram(y=signal, sr=fs))
    fig = plt.figure(figsize=(80, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(log_mel, cmap='rainbow')
    ax1.invert_yaxis()
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(crowd['crowd'].values)
    plt.savefig(f'{output_folder}/logmel_crowd.png')
    plt.close()
    plt.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['plot', 'feature'], default='plot')
    parser.add_argument('-i', '--input-folder', type=str)
    parser.add_argument('-o', '--output-folder', type=str)
    parser.add_argument('-s', '--duration', type=float, default=1.0)
    args = parser.parse_args()
    if args.option == 'plot':
        sim_plot(args.input_folder, args.output_folder)
    elif args.option == 'feature':
        calculate_log_mel_feature(args.input_folder, args.output_folder, args.duration, args.duration)
        # calculate_feature_old(args.input_folder, args.output_folder)
    pass
