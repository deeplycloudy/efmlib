import numpy as np
import pandas as pd

from scipy.optimize import curve_fit


#####
# Fitting functions for normalizing accelerometer, mag field data
#####

def freq_peak(a, fs, skip_dc=3):
    """ Given data in a and sampling frequcy fs in Hz, return the
        frequency (Hz) of the maximum amplitude and the phase (rad) at that frequency
    """
    N = a.shape[0]
    fft_a = np.fft.fft(a)
    freqs = np.fft.fftfreq(N, 1.0/fs)

    if N % 2 == 0:
        # Even
        N_mid = int(N/2)
    else:
        N_mid = int((N-1)/2)

    max_idx = np.argmax(np.abs(fft_a[skip_dc:N_mid]))

    # Get the max frequency from the pos freq half of the FFT
    max_freq = freqs[skip_dc:N_mid][max_idx]
    max_phase = np.angle(fft_a[skip_dc:N_mid][max_idx])
    return max_freq, max_phase

def cos_prototype(x, A, f, phi):
    return A * np.cos(2*np.pi*f*x + phi)

def gen_overlap_chunks(a, chunk_overlap):
    n_chunks = int(np.ceil(a.shape[0]/chunk_overlap))
    for i in range(n_chunks):
        i0 = i*chunk_overlap
        i1 = min(i0+chunk_overlap*2, a.shape[0]+1)
        sl = slice(i0, i1)
        yield a[sl], sl
    assert i1 == a.shape[0]+1


def cosfit(a, interval, fs, guess_amplitude=10.0, unit_amplitude=False):
    """ Given data a, fitting interval in seconds, and sample frequency fs in Hz,
    fit a cosine to overlapping chunks of data.

    The actual fitting interval is chosen so that half of the interval overlaps
    with the next chunk of data on an even array index boundary.

    guess is the [amplitude (units of a), frequency (Hz), phase (rad)] to use
        for the first guess when fitting.

    returns a dictionary with the fit data, and the number of samples in the overlapa

    if unit_amplitude is True, then return a wave with amplitude=1.0; the actual
    fit amplitudes are still returned.

    """
    overlap = interval / 2
    # Get the number of overlapping intervals, and
    overlap_idx = int(overlap * fs)
    interval_idx = int(overlap_idx*2)
    print("Actual interval to match sampling rate is {1:3.2f} s for given {0} s".format(interval, interval_idx / fs))

    accum = np.zeros_like(a)
    fit_A = np.zeros_like(a)
    fit_freq = np.zeros_like(a)
    fit_phase = np.zeros_like(a)
    for chunk, chunk_sl in gen_overlap_chunks(a, overlap_idx):
        t = np.arange(chunk.shape[0])/fs

        guess_freq, guess_phase = freq_peak(chunk, fs)
        if guess_freq < 0: raise

        guess = [guess_amplitude, guess_freq, guess_phase ]
        try:
            params, params_covariance = curve_fit(cos_prototype, t, chunk, p0=guess)
        except RuntimeError as e:
            # Typically, due to non-convergence of the least-squares fitting process.
            print(e)
            import matplotlib.pyplot as plt
            # Make a diagnostic plot
            fig, axes = plt.subplots(1, figsize=(12,6), sharex=True)
            axes.plot(t, chunk, '-k')
            axes.set_title('unable to fit these data with a cosine')
            raise

        fit_chunk = cos_prototype(t, params[0], params[1], params[2])
        if unit_amplitude is True:
            accum[chunk_sl] += fit_chunk/params[0]
        else:
            accum[chunk_sl] += fit_chunk
        fit_A[chunk_sl] += params[0]
        fit_freq[chunk_sl] += params[1]
        fit_phase[chunk_sl] += params[2]

    for d in (accum, fit_A, fit_freq, fit_phase):
        # Not overlapped at start or end
        d[:overlap_idx] *= 2.0
        # Then divide the whole array since it was doubled by overlaps
        d /= 2.0

    return dict(overlap_idx = overlap_idx, fitdata=accum, amplitude=fit_A, phase=fit_phase, frequency=fit_freq)