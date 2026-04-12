import subprocess
import os.path

import librosa
import pandas as pd
# from librosa.feature import melspectrogram
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy as sp
import cv2
from scipy.io import wavfile

FS = 16000


def change_speed_ffmpeg(input_path, output_path, speed=2.0):
    if speed <= 0:
        raise ValueError("speed must be > 0")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise ValueError("Failed to read input FPS")

    out_fps = fps * speed

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-filter:v", f"setpts=PTS/{speed}",
        "-r", str(out_fps),
        "-an",
        output_path
    ]
    subprocess.run(cmd, check=True)


def save_logmel_spec(wav_file, img_file):
    fs, signal = wavfile.read(wav_file)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    if fs != FS:
        signal = librosa.resample(y=signal.astype(float), orig_sr=fs, target_sr=FS)
    raise NotImplemented('save_logmel_spec is not implemented yet')
    # TODO pytorchに移行してmelspectrogramを実装
    # mel = melspectrogram(y=signal, sr=FS)
    # fig, ax = plt.subplots()
    # a = ax.imshow(np.log10(mel), cmap='rainbow')
    # ax.invert_yaxis()
    # fig.colorbar(a, ax=ax, label='Intensity[dB]')
    # plt.savefig(img_file)
    # plt.close()
    # return


def save_logmel_spec2(wav_file, img_file):
    fs, signal = wavfile.read(wav_file)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    if fs != FS:
        signal = librosa.resample(y=signal.astype(float), orig_sr=fs, target_sr=FS)
    folder = os.path.dirname(img_file)
    os.makedirs(folder, exist_ok=True)
    # TODO pytorchに移行してmelspectrogramを実装
    # mel = melspectrogram(y=signal.astype(float), sr=FS) + 1.0e-20
    # plt.figure(figsize=(18, 6))
    # librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), x_axis='time', y_axis='mel', sr=FS,
    #                          cmap='rainbow')
    # cbar = plt.colorbar(format='%+2.0f dB')
    # cbar.ax.tick_params(labelsize=16)
    # plt.title('Mel Spectrogram', fontsize=24)
    # plt.xlabel('Time [s]', fontsize=20)
    # plt.ylabel('Frequency [Hz]', fontsize=20)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.tight_layout()
    # plt.savefig(img_file)
    # plt.close()
    return


def save_spectrogram(wav_file, img_file):
    fs, signal = wavfile.read(wav_file)
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    if fs != FS:
        signal = librosa.resample(y=signal.astype(float), orig_sr=fs, target_sr=FS)
    folder = os.path.dirname(img_file)
    os.makedirs(folder, exist_ok=True)
    _, _, stft = sp.signal.stft(x=signal.astype(float), fs=FS, window='hann', nperseg=1024,
                                noverlap=1024 - 256)
    plt.figure(figsize=(18, 6))
    # plt.imshow(np.log10(np.abs(stft) + 1.0e-6)*20, cmap='rainbow', aspect='auto', origin='lower')
    plt.imshow(np.log10((np.abs(stft) + 1.0e-6) / (np.max(np.abs(stft)) + 1.0e-6))*20,
               cmap='jet', aspect='auto', origin='lower')
    # plt.imshow(np.abs(stft) ** 0.2, cmap='jet', aspect='auto', origin='lower')
    plt.colorbar(label='Intensity[dBFS]')
    plt.title('Spectrogram', fontsize=24)
    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Frequency [Hz]', fontsize=20)
    # プロットされた図のx軸を秒表示にする
    x_step = int(stft.shape[1] / 10)
    plt.xticks(ticks=np.arange(0, stft.shape[1], step=x_step),
               labels=[f'{t * 256 / FS:.1f}' for t in np.arange(0, stft.shape[1], step=x_step)],
               fontsize=16)
    # プロットされた図のy軸を周波数表示にする
    plt.yticks(ticks=np.arange(0, stft.shape[0], step=128),
               labels=[f'{f * FS / 1024:.0f}' for f in np.arange(0, stft.shape[0], step=128)],
               fontsize=16)
    # plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(img_file)
    plt.close()
    return


def save_scatter(csv_file, img_file, xlabel, ylabel, title, x_col, y_col):
    df = pd.read_csv(csv_file)
    x, y = df[x_col].values, df[y_col].values
    val_min, val_max = min([x.min(), y.min()]), max([x.max(), y.max()])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, color='b', marker='.', alpha=0.3)
    ax.plot([val_min, val_max], [val_min, val_max], '--')
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid()
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    folder = os.path.dirname(img_file)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(img_file)
    plt.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, default='logmel',
                        choices=['logmel', 'scatter', 'stft'])
    parser.add_argument('-i', '--input-file', type=str)
    parser.add_argument('-o', '--output-file', type=str)
    parser.add_argument('-xl', '--xlabel', type=str, nargs='*',
                        default=['Actual', '[people/s]'])
    parser.add_argument('-yl', '--ylabel', type=str, nargs='*',
                        default=['Estimated', '[people/s]'])
    parser.add_argument('-xc', '--x_col', type=str)
    parser.add_argument('-yc', '--y_col', type=str)
    parser.add_argument('-gt', '--title', type=str, nargs='*')
    args = parser.parse_args()
    if args.option == 'logmel':
        save_logmel_spec2(args.input_file, args.output_file)
    elif args.option == 'stft':
        save_spectrogram(args.input_file, args.output_file)
    elif args.option == 'scatter':
        save_scatter(args.input_file, args.output_file,
                     xlabel=' '.join(args.xlabel), ylabel=' '.join(args.ylabel),
                     x_col=args.x_col, y_col=args.y_col, title=' '.join(args.title))
