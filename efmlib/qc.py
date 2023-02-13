import numpy as np
import pandas as pd

def df_fiber_qc(df_fiber, data_period=22.5, max_milli=2):
    """
    Data recorded by the EFM has many many artifacts in the output.  
    These are likely due to the crap connection on the fiber at the rotating 
    interface.  The fiber here is just a butt connection, and one side of it 
    is rotating.  Not something that exactly meets the quality standards of 
    Norman building code.  So, we need to go through the data pretty carefully 
    looking for crap that isn't real.

    This will mean that later on, we don't get monotonically spaced time.  
    Hope no one needs to do any Fourier analysis
    """

    # first to examine is time.  This field is stored in the obviously named
    # 'adc_ready_millis' field
    # what unit is millis exactly Eric?  Why do they need to be ready?  Who knows

    # The most consistent way I've found to filter time is to calculate the 
    # average period, and make sure it's within expected bounds
    # we can't check for it to be over a lower bound, because the data recording 
    # has a tendency to restart part way through the flight
    df_fiber['period'] = df_fiber['adc_ready_millis']/np.arange( len(df_fiber['adc_ready_millis']) )
    df_fiber = df_fiber[ (df_fiber['period']<data_period)]


    # extra filter: get rid of repeated, lines
    fields = ['acceleration_x', 'acceleration_y', 'acceleration_z',
              'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
              'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    df_fiber = df_fiber[(df_fiber.diff()[fields] == 0).sum(axis=1)<6]

    # get rid of lines where the adc reading is out of range
    # the EFM voltage is read off of a 24 bit adc, then read into a 32 bit field
    # so there's a lot of room for bunk values
    # the test is against 2**23 because the value is signed
    df_fiber = df_fiber[np.abs(df_fiber['adc_reading'])<2**23]  

    # df_fiber = df_fiber[np.abs((df_fiber.diff()['adc_ready_millis']))<1e4]
    return df_fiber

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