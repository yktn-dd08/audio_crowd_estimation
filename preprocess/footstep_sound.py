import os.path

import librosa
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydantic import BaseModel


WAV_FILE_FOOTSTEP = [
    '1-155858-A-25.wav',
    '1-155858-B-25.wav',
    '1-155858-C-25.wav',
    '1-155858-D-25.wav',
    '1-155858-E-25.wav',
    '1-155858-F-25.wav',
    '1-223162-A-25.wav',
    '1-51147-A-25.wav',
    '2-209471-A-25.wav',
    '2-209472-A-25.wav',
    '2-209473-A-25.wav',
    '2-209474-A-25.wav',
    '2-209475-A-25.wav',
    '2-209476-A-25.wav',
    '2-209477-A-25.wav',
    '2-209478-A-25.wav',
    '3-103597-A-25.wav',
    '3-103598-A-25.wav',
    '3-103599-A-25.wav',
    '3-103599-B-25.wav',
    '3-249913-A-25.wav',
    '3-94342-A-25.wav',
    '3-94343-A-25.wav',
    '3-94344-A-25.wav',
    '4-117627-A-25.wav',
    '4-117630-A-25.wav',
    '4-194979-A-25.wav',
    '4-194981-A-25.wav',
    '4-198962-A-25.wav',
    '4-198962-B-25.wav',
    '4-218304-A-25.wav',
    '4-218304-B-25.wav',
    '5-234263-A-25.wav',
    '5-234855-A-25.wav',
    '5-243025-A-25.wav',
    '5-244310-A-25.wav',
    '5-263490-A-25.wav',
    '5-263491-A-25.wav',
    '5-263501-A-25.wav',
    '5-51149-A-25.wav'
]
FS = 16000


class EnvSoundInfo(BaseModel):
    file: str
    segment: list[float]
    type: str | None = None
    flag: bool = True

    def save_segment(self, org_folder, res_folder, normalize=True, base_position=0.1):
        fs, signal = wavfile.read(f'{org_folder}/{self.file}')
        for i in range(len(self.segment) - 1):
            pass
        return


class EnvSoundConfig(BaseModel):
    config: dict[str, EnvSoundInfo]


def plot_spectrogram():
    folder = './data/ambient_sound/audio'
    for wf in WAV_FILE_FOOTSTEP:
        fs, signal = wavfile.read(f'{folder}/{wf}')
        if fs != FS:
            signal = librosa.resample(y=signal.astype(float), orig_sr=fs, target_sr=FS)

        # fig = plt.figure()
        _, _, stft = sp.signal.stft(x=signal, fs=FS, window='hann', nperseg=1024,
                                    noverlap=1024 - 256)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(np.abs(stft)**0.1, cmap='rainbow')
        ax.invert_yaxis()
        plt.xlabel("Time index")
        plt.ylabel("Frequency index")
        output_folder = 'workspace/footstep'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(f'{output_folder}/' + wf.replace('.wav', '.png'))
        plt.close()
        plt.cla()

        power_series = np.sum(np.abs(stft), axis=0)
        plt.figure()
        plt.plot(np.log10(power_series))
        plt.savefig(f'{output_folder}/pt_' + wf.replace('.wav', '.png'))
        plt.close()
        plt.cla()
    return


def footstep_decompose(wav_name):
    fs, signal = wavfile.read(wav_name)
    if fs != FS:
        # down-sampling
        signal = librosa.resample(y=signal.astype(float), orig_sr=fs, target_sr=FS)

    return


if __name__ == '__main__':
    # TODO implement of footstep decomposition
    plot_spectrogram()
