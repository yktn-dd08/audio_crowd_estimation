"""Documented MBSS implementation for AFPILD; raw recordings are never modified.

AFPILD specifies MBSS and an initial approximately 3-second noise interval, but
not its coefficients. Here the equations/weights in Kamath & Loizou (2002),
Eqs. 4--9, provide explicit implementation choices; this is not a claim to
reproduce the unpublished AFPILD denoiser exactly. No output is peak normalized.
"""
import numpy as np
from scipy import signal, ndimage

FS = 16000
WINDOW = 320
HOP = 160
NFFT = 512
WEIGHTS = np.array([.09, .25, .32, .25, .09])
BANDS = np.array([0., 2000., 4000., 6000., 8000.])
BETA = .002
EPS = 1e-20


def transform(y):
    return signal.stft(y, FS, window='hamming', nperseg=WINDOW,
                       noverlap=WINDOW-HOP, nfft=NFFT, boundary='zeros', padded=True)


def invert(z, length):
    _, y = signal.istft(z, FS, window='hamming', nperseg=WINDOW,
                        noverlap=WINDOW-HOP, nfft=NFFT, boundary=True)
    return y[:length]


def noise_frames(prefix, first_event_s, mode='quiet'):
    """Shared frame selection for four channels; never inspect simulated audio."""
    _, times, _ = transform(prefix[:, 0])
    end = min(3., first_event_s-.2) if mode == 'quiet' else 3.
    valid = np.flatnonzero((times >= WINDOW/(2*FS)) & (times <= end-WINDOW/(2*FS)))
    assert len(valid) >= 20
    energies = []
    for ch in range(prefix.shape[1]):
        _, _, z = transform(prefix[:, ch])
        energies.append((np.abs(z[:, valid])**2).sum(axis=0))
    energy = np.mean(energies, axis=0)
    if mode == 'quiet':
        valid = np.sort(valid[np.argsort(energy)[:max(20, int(np.ceil(.3*len(valid))))]])
    return valid, {'mode': mode, 'candidate_end_s': end,
                   'selected_frame_times_s': times[valid].tolist(),
                   'selected_frames': len(valid),
                   'actual_selected_span_s': [float(times[valid[0]]-.01), float(times[valid[-1]]+.01)]}


def estimate_noise(prefix, frame_indices):
    spectra = []
    for ch in range(prefix.shape[1]):
        _, _, z = transform(prefix[:, ch])
        spectra.append(np.mean(np.abs(z[:, frame_indices])**2, axis=1))
    return np.array(spectra)


def subtract(z, noise_power):
    """Power subtraction with per-band SNR coefficients and original phase.

Implements the published five-frame magnitude average. The separately cited
prior spectral smoother in the paper is not specified/reimplemented here.
The beta floor applies only when subtraction produces a nonpositive value,
following Eq. 7; it is relative to the (smoothed) noisy power, not noise power.
"""
    amplitude = np.abs(z)
    smoothed = ndimage.convolve1d(amplitude, WEIGHTS, axis=1, mode='nearest')
    power = smoothed**2
    clean = np.empty_like(power)
    freq = np.fft.rfftfreq(NFFT, 1/FS)
    alphas = []
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        mask = (freq >= low) & ((freq < high) if high < FS/2 else (freq <= high))
        snr = 10*np.log10((power[mask].sum(axis=0)+EPS)/(noise_power[mask].sum()+EPS))
        alpha = np.where(snr < -5., 5., np.where(snr > 20., 1., 4.-.15*snr))
        delta = 1. if high <= 1000 else (2.5 if high <= FS/2-2000 else 1.5)
        estimate = power[mask] - alpha[None, :]*delta*noise_power[mask, None]
        clean[mask] = np.where(estimate > 0., estimate, BETA*power[mask])
        alphas.append([float(alpha.mean()), float(np.min(alpha)), float(np.max(alpha))])
    phase = z / np.maximum(amplitude, EPS)
    return np.sqrt(clean)*phase, {'alpha_mean_min_max_by_band': alphas,
                                'floored_bin_fraction': float(np.mean(clean == BETA*power))}


def denoise(y, noise_power):
    _, _, z = transform(y)
    enhanced, diagnostics = subtract(z, noise_power)
    result = invert(enhanced, len(y))
    assert len(result) == len(y) and np.isfinite(result).all()
    return result, diagnostics
