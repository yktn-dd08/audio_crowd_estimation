import librosa
from librosa.feature import melspectrogram
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy as sp
from scipy.io import wavfile


FS = 16000


def save_logmel_spec(wav_file, img_file):
    fs, signal = wavfile.read(wav_file)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    if fs != FS:
        signal = librosa.resample(y=signal.astype(float), orig_sr=fs, target_sr=FS)
    mel = melspectrogram(y=signal, sr=FS)
    fig, ax = plt.subplots()
    a = ax.imshow(np.log10(mel), cmap='rainbow')
    ax.invert_yaxis()
    fig.colorbar(a, ax=ax, label='Intensity[dB]')
    plt.savefig(img_file)
    plt.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, default='logmel',
                        choices=['logmel'])
    parser.add_argument('-i', '--input-file', type=str)
    parser.add_argument('-o', '--output-file', type=str)
    args = parser.parse_args()
    if args.option == 'logmel':
        save_logmel_spec(args.input_file, args.output_file)
