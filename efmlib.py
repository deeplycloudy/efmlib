import struct

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit


#####
# I/O functions
#####

valid_ranges = dict(
    adc_volts = (-5, 5),
    acceleration_x = (-25, 25),
    acceleration_y = (-25, 25),
    acceleration_z = (-25, 25),
    magnetometer_x = (-25,125),
    magnetometer_y = (-75, 75),
    magnetometer_z = (-60, 60),
    gyroscope_x = (-10, 10),
    gyroscope_y = (-10, 10),
    gyroscope_z = (-5, 5),
)

def decode_gps_packet(mp):
    result = dict()
    result['start_byte'] = struct.unpack('B', mp[0:1])[0]
    result['millis'] = struct.unpack('I', mp[1:5])[0]
    result['latitude'] = struct.unpack('i', mp[5:9])[0] / 10000
    result['longitude'] = struct.unpack('i', mp[9:13])[0] / 10000
    result['altitude'] = struct.unpack('H', mp[13:15])[0]
    result['gps_time'] = struct.unpack('I', mp[15:19])[0]
    result['end_byte'] = struct.unpack('B', mp[19:])[0]
    return result

def decode_data_packet(mp):
    result = dict()
    result['start_byte'] = struct.unpack('B', mp[0:1])[0]
    result['adc_ready_millis'] = struct.unpack('I', mp[1:5])[0]
    result['adc_reading'] = struct.unpack('i', mp[5:9])[0]
    result['acceleration_x'] = struct.unpack('f', mp[9:13])[0]
    result['acceleration_y'] = struct.unpack('f', mp[13:17])[0]
    result['acceleration_z'] = struct.unpack('f', mp[17:21])[0]
    result['magnetometer_x'] = struct.unpack('f', mp[21:25])[0]
    result['magnetometer_y'] = struct.unpack('f', mp[25:29])[0]
    result['magnetometer_z'] = struct.unpack('f', mp[29:33])[0]
    result['gyroscope_x'] = struct.unpack('f', mp[33:37])[0]
    result['gyroscope_y'] = struct.unpack('f', mp[37:41])[0]
    result['gyroscope_z'] = struct.unpack('f', mp[41:45])[0]
    result['temperature'] = struct.unpack('H', mp[45:47])[0] / 10
    result['relative_humidity'] = struct.unpack('H', mp[47:49])[0]
    result['pressure'] = struct.unpack('H', mp[49:51])[0] / 10
    result['end_byte'] = struct.unpack('B', mp[51:])[0]
    return result

def fix_adc_offset(df, dt=0.065, up=50, adc_var='adc_volts_withlag'):
    """
    The charge amplifier output exhibits a lag relative to the IMU, probably
    beacuse of some combination of an internal filter in the ADC and a phase
    offset produced by the charge amplifier.

    So, this function moves the adc_volts signal in df earlier in time by dt.
    Upsamples by up before shifting a fixed number of data points equivalent to dt.
    Returns a pandas Series with the same shape as df.adc_volts_withlag.

    % Need to advance the charge amplifier signal by dt to compensate for time
    % delay.  To do this, interpolate by a fairly large amount, then shift
    % charge amplifier data by an amount corresponding to the time delay
    %NI = 200;   % Number to up-sample by
    NI = 50;    % Number to up-sample by (Time resolution = 0.71 ms)
    E = interp(e0_raw-mean(e0_raw),NI);
    new_fs = fs*NI;
    dn = round(new_fs*dt);   % Number of samples to advance
    E=E(dn:NI:length(E));
    % Need to add one or two samples to end to make vo right length
    E(length(E):length(ar_raw))=0;
    fprintf(['E data upsampled, shifted, backsampled ... \n'])
    """

    from scipy.signal import resample

    t = df.adc_ready_millis*1.0e-3 # seconds
    delta_ts = t.diff()
    fs = 1.0/delta_ts.median()
    new_fs = fs*up

    n_orig = df[adc_var].shape[0]
    n_up = n_orig*up
    adc_up = resample(df[adc_var], n_up)
    dn = int(np.floor(new_fs*dt)) # Number of samples to advance
    print("Shifting ADC backward by {0:3.1f} original samples.".format(new_fs*dt/up))
    new_adc = adc_up[dn::up]
    n_pad = n_orig - new_adc.shape[0]
    adc_volts_shifted = pd.Series(
        np.hstack([new_adc, np.zeros(n_pad)])
    )
    assert adc_volts_shifted.shape[0] == n_orig
    return adc_volts_shifted


def read_efm_raw(filenames, shift_dt=0.065):
    ba = bytearray()
    for filename in filenames:
        with open(filename, 'rb') as f:
            ba = ba + f.read()

    data_start_bytes = []
    gps_start_bytes = []
    data_packet_length = 51
    gps_packet_length = 19

    # Determine the valid starting bytes for data and gps packets
    for i in range(len(ba) - data_packet_length):
        if (ba[i] == 190) and (ba[i+data_packet_length] == 239):
            data_start_bytes.append(i)

    for i in range(len(ba) - gps_packet_length):
        if (ba[i] == 254) and (ba[i+gps_packet_length] == 237):
            #print(ba[i], ba[i+data_packet_length])
            gps_start_bytes.append(i)

    gps_raw_packets = []
    data_raw_packets = []

    gps_packets = []
    data_packets = []

    for sb in gps_start_bytes:
        gps_raw_packets.append(ba[sb:sb+gps_packet_length+1])
        gps_packets.append(decode_gps_packet(ba[sb:sb+gps_packet_length+1]))

    for sb in data_start_bytes:
        data_raw_packets.append(ba[sb:sb+data_packet_length+1])
        data_packets.append(decode_data_packet(ba[sb:sb+data_packet_length+1]))

    adc_cal = (2 * 2.048) / 2**24
    adc_cal *= 2 # For voltage divider that is in-place

    series = {}
    for field in data_packets[0].keys():
        vals = []
        for p in data_packets:
            vals.append(p[field])
        series[field] = vals
    df_fiber = pd.DataFrame(series)
    df_fiber['adc_volts_withlag'] = adc_cal * df_fiber['adc_reading']
    df_fiber['adc_volts'] = fix_adc_offset(df_fiber, dt=shift_dt)

    series = {}
    for field in gps_packets[0].keys():
        vals = []
        for p in gps_packets:
            vals.append(p[field])
        series[field] = vals
    df_gps = pd.DataFrame(series)
    df_gps['gps_time'] =  pd.to_datetime(df_gps['gps_time'], format='%H%M%S00', errors='coerce')
    df_gps.replace([0], np.nan, inplace=True)

    return df_fiber, df_gps


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