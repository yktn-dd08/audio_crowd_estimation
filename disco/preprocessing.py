import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


FRAME_SIZE = 2048
FRAME_SHIFT = 512


def read_files(mat_file, wav_file, folder):
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

    # calculate audio features
    ace_data = {'crowd': int(mat_data['map'].sum().round())}
    _, _, spec = sp.signal.stft(x=wav_data.mean(axis=1), fs=fs, window='hann', nperseg=FRAME_SIZE,
                                noverlap=FRAME_SIZE - FRAME_SHIFT)
    spec = np.abs(spec) ** 2
    power = float(spec.sum())

    return


if __name__ == '__main__':
    print()
