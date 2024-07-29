import os
import json
import glob
import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import py2shpss as hpss

from librosa.feature import mfcc, melspectrogram
from tqdm import tqdm
from joblib import Parallel, delayed


FRAME_SIZE = 2048
FRAME_SHIFT = 512
"""
mat_file = './data/disco/density_map/0007.mat'
wav_file = './data/disco/audio/0007.wav'
"""


def nmf_decomposition(signal, k=10, h=None, u=None):

    return


def stats_analysis(signal, fs, label=None):
    tmp = dict()
    tmp['kurtosis'] = sp.stats.kurtosis(signal)
    _, _, stft = sp.signal.stft(x=signal, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                noverlap=FRAME_SIZE - FRAME_SHIFT)
    tmp['stft_log_power'] = np.log10((np.abs(stft)**2).sum())

    stats = {k if label is None else f'{label}_{k}': v for k, v in tmp.items()}
    return stats


def feature_analysis(wav_file, folder):
    feature = {}
    fs, wav_data = sp.io.wavfile.read(wav_file)
    signal = wav_data.mean(axis=1)
    _, _, stft = sp.signal.stft(x=signal, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                noverlap=FRAME_SIZE - FRAME_SHIFT)
    feature['logmel'] = np.log10(melspectrogram(y=signal, sr=fs))
    feature['stft'] = stft
    feature['stft_power'] = np.log10((np.abs(stft)**2).sum())
    feature['kurtosis'] = sp.stats.kurtosis(signal)

    # hpss
    hpss2s = hpss.twostageHPSS(samprate=fs)
    ll = len(signal)
    ms = np.max(signal)
    h, v, p = hpss2s(signal/ms)
    h = h[0:ll] * ms if len(h) > ll else np.concatenate([h * ms, np.zeros(ll - len(h))])
    v = v[0:ll] * ms if len(v) > ll else np.concatenate([v * ms, np.zeros(ll - len(v))])
    p = p[0:ll] * ms if len(p) > ll else np.concatenate([p * ms, np.zeros(ll - len(p))])
    
    feature['h_logmel'] = np.log10(melspectrogram(y=h, sr=fs))
    feature['v_logmel'] = np.log10(melspectrogram(y=v, sr=fs))
    feature['p_logmel'] = np.log10(melspectrogram(y=p, sr=fs))
    _, _, h_stft = sp.signal.stft(x=h, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                  noverlap=FRAME_SIZE - FRAME_SHIFT)
    feature['h_stft'] = h_stft
    feature['h_stft_power'] = np.log10((np.abs(h_stft)**2).sum())
    feature['h_kurtosis'] = sp.stats.kurtosis(h)

    _, _, v_stft = sp.signal.stft(x=v, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                  noverlap=FRAME_SIZE - FRAME_SHIFT)
    feature['v_stft'] = v_stft
    feature['v_stft_power'] = np.log10((np.abs(v_stft)**2).sum())
    feature['v_kurtosis'] = sp.stats.kurtosis(v)

    _, _, p_stft = sp.signal.stft(x=p, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                  noverlap=FRAME_SIZE - FRAME_SHIFT)
    feature['p_stft'] = p_stft
    feature['p_stft_power'] = np.log10((np.abs(p_stft)**2).sum())
    feature['p_kurtosis'] = sp.stats.kurtosis(p)

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(f'{folder}/feature.pickle', mode='wb') as f:
        pickle.dump(feature, f)
    return


def save_log_melspec(signal, fs, save_path):
    mel = melspectrogram(y=signal, sr=fs)
    fig, ax = plt.subplots()
    a = ax.imshow(np.log10(mel), cmap='rainbow')
    ax.invert_yaxis()
    fig.colorbar(a, ax=ax, label='Intensity[dB]')
    plt.savefig(save_path)
    plt.close()
    return


def preliminary_analysis(mat_file, jpg_file, wav_file, folder):
    """
    Create spectrogram, people flow image, etc
    :param mat_file:
    :param jpg_file:
    :param wav_file:
    :param folder:
    :return:
    """
    mat_data = sp.io.loadmat(mat_file)
    fs, wav_data = sp.io.wavfile.read(wav_file)
    signal = wav_data.mean(axis=1)

    if not os.path.exists(folder):
        os.makedirs(folder)

    # map image
    plt.imshow(mat_data['map'])
    plt.savefig(f'{folder}/density_map.png')
    plt.close()

    # spectrogram
    fig = plt.figure(figsize=(10, 4))
    spectrogram, freq, t, im = plt.specgram(signal, NFFT=FRAME_SIZE, noverlap=FRAME_SIZE - FRAME_SHIFT,
                                            Fs=fs)
    fig.colorbar(im).set_label('Intensity [dB]')
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.savefig(f'{folder}/spec.png')
    plt.close()

    img = plt.imread(jpg_file, format='jpg')
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(img)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.specgram(signal, NFFT=FRAME_SIZE, noverlap=FRAME_SIZE - FRAME_SHIFT, Fs=fs)
    plt.savefig(f'{folder}/merge.png')
    plt.close()

    # log mel spectrogram
    save_log_melspec(signal, fs, f'{folder}/log-melspectrogram.png')

    # hpss
    hpss2s = hpss.twostageHPSS(samprate=fs)
    h, v, p = hpss2s(signal/np.max(signal))

    save_log_melspec(h*np.max(signal), fs, f'{folder}/h-log-melspectrogram.png')
    save_log_melspec(v*np.max(signal), fs, f'{folder}/v-log-melspectrogram.png')
    save_log_melspec(p*np.max(signal), fs, f'{folder}/p-log-melspectrogram.png')
    sp.io.wavfile.write(f'{folder}/h.wav', fs, (h*np.max(signal)).astype(np.int16))
    sp.io.wavfile.write(f'{folder}/v.wav', fs, (v*np.max(signal)).astype(np.int16))
    sp.io.wavfile.write(f'{folder}/p.wav', fs, (p*np.max(signal)).astype(np.int16))

    # calculate audio features
    ace_data = {'crowd': int(mat_data['map'].sum().round())}
    _, _, spec = sp.signal.stft(x=signal, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                noverlap=FRAME_SIZE - FRAME_SHIFT)
    spec = np.abs(spec) ** 2
    ace_data['power'] = float(np.log10(spec.sum()))
    mfcc_feat = mfcc(y=wav_data.mean(axis=1), sr=fs)
    # ace_data['mfcc_avg'] = mfcc_feat.mean(axis=1).tolist()
    # ace_data['mfcc_std'] = mfcc_feat.std(axis=1).tolist()
    ace_data['kurtosis'] = sp.stats.kurtosis(wav_data.mean(axis=1))
    multi_hpss = hpss.twostageHPSS(samprate=fs)

    with open(f'{folder}/simple_stats.json', 'w') as f:
        json.dump(ace_data, f)
    return


def feature_calculation(disco_folder, output_folder):
    wav_list = glob.glob(f'{disco_folder}/audio/*.wav')
    sorted(wav_list)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    parallel_list = []
    for wav_file in wav_list:
        base_name = os.path.basename(wav_file)
        folder = f'{output_folder}/' + base_name.replace('.wav', '')
        parallel_list.append([wav_file, folder])

    Parallel(n_jobs=-1)(delayed(feature_analysis)(p[0], p[1]) for p in tqdm(parallel_list))
    return


def preprocess(disco_folder, output_folder):
    mat_list = glob.glob(f'{disco_folder}/density_map/*.mat')
    sorted(mat_list)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    parallel_list = []
    for mat_file in mat_list:
        base_name = os.path.basename(mat_file)
        jpg_file = f'{disco_folder}/image/' + base_name.replace('.mat', '.jpg')
        wav_file = f'{disco_folder}/audio/' + base_name.replace('.mat', '.wav')
        folder = f'{output_folder}/' + base_name.replace('.mat', '')
        parallel_list.append([mat_file, jpg_file, wav_file, folder])

    Parallel(n_jobs=-1)(delayed(preliminary_analysis)(p[0], p[1], p[2], p[3]) for p in tqdm(parallel_list))
    # preliminary_analysis(mat_file, jpg_file, wav_file, folder)
    return


def read_merged_data(folder, verbose=True):
    folder_list = glob.glob(f'{folder}/*')
    sorted(folder_list)
    data = []
    for i, fl in tqdm(enumerate(folder_list), disable=not verbose):
        pickle_path = f'{fl}/feature.pickle'
        json_path = f'{fl}/simple_stats.json'
        if os.path.exists(pickle_path) and os.path.exists(json_path):
            with open(pickle_path, 'rb') as f:
                feature = pickle.load(f)
            with open(json_path, 'r') as f:
                stats_js = json.load(f)
            data.append(feature | stats_js | {'folder': fl})

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--disco-folder')
    parser.add_argument('-o', '--output-folder')
    parser.add_argument('-opt', '--option', choices=['preprocess', 'feature'],
                        default='preprocess')
    args = parser.parse_args()
    if args.option == 'preprocess':
        preprocess(args.disco_folder, args.output_folder)
    elif args.option == 'feature':
        feature_calculation(args.disco_folder, args.output_folder)
