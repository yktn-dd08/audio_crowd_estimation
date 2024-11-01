import argparse
import os.path
import numpy as np
import scipy as sp
from scipy.io import wavfile
from fast_bss_eval import sdr
from common.logger import get_logger

FRAME_SIZE = 1024
FRAME_SHIFT = 512
EPS = 1e-35
logger = get_logger('tools.denoise')


def spectral_subtraction(input_file, output_file, noise_end, noise_start=0.0, alpha=2.0, p=1.0):
    logger.info(f'denoise - spectral subtraction - alpha: {alpha}, p: {p}, noise time: [{noise_start}, {noise_end}]')
    fs, signal = wavfile.read(input_file)
    f, t, stft = sp.signal.stft(x=signal, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                noverlap=FRAME_SIZE - FRAME_SHIFT)
    stft_amp = np.abs(stft)
    phase = stft / np.maximum(stft_amp, EPS)
    noise_frame_start = np.sum(t < noise_start / fs)
    noise_frame_end = np.sum(t < noise_end / fs)
    assert noise_frame_start < noise_frame_end, f'Invalid noise frame : {noise_frame_start} - {noise_frame_end}'

    noise_amp = np.mean(stft_amp[:, noise_frame_start:noise_frame_end] ** p, axis=1, keepdims=True) ** 1 / p
    stft_eps = 0.01 * stft_amp

    denoise_amp = np.maximum(stft_amp ** p - alpha * noise_amp ** p, stft_eps ** p) ** 1 / p
    _, denoise_signal = sp.signal.istft(denoise_amp * phase, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                        noverlap=FRAME_SIZE - FRAME_SHIFT)
    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    wavfile.write(output_file, fs, denoise_signal/denoise_signal.max())
    snr_list = sdr(denoise_signal[None, :], signal[None, :])
    logger.info(f'denoise result - snr: {snr_list[0]}')
    return


def wiener_filter(input_file, output_file, noise_end, noise_start=0.0, alpha=1.0, mu=10):
    logger.info(f'denoise - wiener fileter - alpha: {alpha}, mu: {mu}, noise time: [{noise_start}, {noise_end}]')
    fs, signal = wavfile.read(input_file)
    f, t, stft = sp.signal.stft(x=signal, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                noverlap=FRAME_SIZE - FRAME_SHIFT)
    stft_amp = np.abs(stft)
    phase = stft / np.maximum(stft_amp, EPS)
    noise_frame_start = np.sum(t < noise_start / fs)
    noise_frame_end = np.sum(t < noise_end / fs)
    assert noise_frame_start < noise_frame_end, f'Invalid noise frame : {noise_frame_start} - {noise_frame_end}'

    noise_power = np.mean(stft_amp[:, noise_frame_start:noise_frame_end] ** 2, axis=1, keepdims=True)
    stft_eps = 0.01 * stft_amp ** 2

    est_power = np.maximum(stft_amp ** 2 - alpha * noise_power, stft_eps)
    w_filter = est_power / (est_power + mu * noise_power)

    denoise_amp = w_filter * stft_amp
    _, denoise_signal = sp.signal.istft(denoise_amp * phase, fs=fs, window='hann', nperseg=FRAME_SIZE,
                                        noverlap=FRAME_SIZE - FRAME_SHIFT)
    output_folder = os.path.dirname(output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    wavfile.write(output_file, fs, denoise_signal / denoise_signal.max())
    snr_list = sdr(denoise_signal[None, :], signal[None, :])
    logger.info(f'denoise result - snr: {snr_list[0]}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, default='ss',
                        choices=['ss', 'wf'])
    parser.add_argument('-i', '--input-file', type=str)
    parser.add_argument('-o', '--output-file', type=str)
    parser.add_argument('-s', '--noise-start', type=float, default=0.0)
    parser.add_argument('-e', '--noise-end', type=float)
    args = parser.parse_args()

    if args.option == 'ss':
        spectral_subtraction(args.input_file, args.output_file, args.noise_end, args.noise_start)
    elif args.option == 'wf':
        wiener_filter(args.input_file, args.output_file, args.noise_end, args.noise_start)
