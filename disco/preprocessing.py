import os
import json
import glob
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from librosa.feature import mfcc
from tqdm import tqdm

FRAME_SIZE = 2048
FRAME_SHIFT = 512
"""
mat_file = './data/disco/density_map/0007.mat'
wav_file = './data/disco/audio/0007.wav'
"""


def calculate_features(mat_file, jpg_file, wav_file, folder):
    mat_data = sp.io.loadmat(mat_file)
    fs, wav_data = sp.io.wavfile.read(wav_file)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # map image
    plt.imshow(mat_data['map'])
    plt.savefig(f'{folder}/density_map.png')
    plt.close()

    # spectrogram
    fig = plt.figure(figsize=(10, 4))
    spectrogram, freq, t, im = plt.specgram(wav_data.mean(axis=1), NFFT=FRAME_SIZE, noverlap=FRAME_SIZE - FRAME_SHIFT,
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
    ax2.specgram(wav_data.mean(axis=1), NFFT=FRAME_SIZE, noverlap=FRAME_SIZE - FRAME_SHIFT,Fs=fs)
    plt.savefig(f'{folder}/merge.png')
    plt.close()

    # calculate audio features
    ace_data = {'crowd': int(mat_data['map'].sum().round())}
    _, _, spec = sp.signal.stft(x=wav_data.mean(axis=1), fs=fs, window='hann', nperseg=FRAME_SIZE,
                                noverlap=FRAME_SIZE - FRAME_SHIFT)
    spec = np.abs(spec) ** 2
    ace_data['power'] = float(np.log10(spec.sum()))
    mfcc_feat = mfcc(y=wav_data.mean(axis=1), sr=fs)
    ace_data['mfcc_avg'] = mfcc_feat.mean(axis=1).tolist()
    ace_data['mfcc_std'] = mfcc_feat.std(axis=1).tolist()
    ace_data['kurtosis'] = sp.stats.kurtosis(wav_data.mean(axis=1))

    with open(f'{folder}/feature.json', 'w') as f:
        json.dump(ace_data, f)
    return


def preprocess(disco_folder, output_folder):
    mat_list = glob.glob(f'{disco_folder}/density_map/*.mat')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for mat_file in tqdm(mat_list):
        base_name = os.path.basename(mat_file)
        jpg_file = f'{disco_folder}/image/' + base_name.replace('.mat', '.jpg')
        wav_file = f'{disco_folder}/audio/' + base_name.replace('.mat', '.wav')
        folder = f'{output_folder}/' + base_name.replace('.mat', '')
        calculate_features(mat_file, jpg_file, wav_file, folder)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--disco-folder')
    parser.add_argument('-o', '--output-folder')
    args = parser.parse_args()
    preprocess(args.disco_folder, args.output_folder)
