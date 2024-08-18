import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import py2shpss as hpss
from tqdm import tqdm
from scipy.io import wavfile
from librosa.feature import melspectrogram


def each_feat_calc(crowd, signal, fs, index):
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
    feat = np.stack([np.log10(melspectrogram(y=d, sr=fs)) for d in [s, h, v, p]])
    return y, feat


def calculate_feature(input_folder, output_folder):
    crowd = pd.read_csv(f'{input_folder}/crowd.csv')
    fs, signal = wavfile.read(f'{input_folder}/sim0.wav')
    xy = [each_feat_calc(crowd, signal, fs, t) for t in tqdm(crowd['crowd'])]
    y = np.array([f[0] for f in xy])
    x = np.stack([f[1] for f in xy])
    feature = {'y': y, 'x': x}
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(f'{output_folder}/feature.pickle', 'wb') as f:
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
    args = parser.parse_args()
    if args.option == 'plot':
        sim_plot(args.input_folder, args.output_folder)
    elif args.option == 'feature':
        calculate_feature(args.input_folder, args.output_folder)
    pass
