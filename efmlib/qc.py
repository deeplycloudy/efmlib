import numpy as np
import pandas as pd
from scipy import optimize

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
    t0 = df_fiber['adc_ready_millis'][:100].median()
    df_fiber['period'] = (df_fiber['adc_ready_millis']-t0)/np.arange( len(df_fiber['adc_ready_millis']) )
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

def cosFn ( data_milli, amplitude, frequency, phase, offset ):
    #force some constraints
    amplitude = abs(amplitude)
    frequency = abs(frequency)

    t0 = np.array(data_milli)[0]
    return  amplitude*np.cos( 2*np.pi*(data_milli-t0)/1000.*frequency + phase ) + offset

def df_fiber_filter( df_fiber, sample_rate=45.45, adc_offset=-0.065 ):
    """
    Data recorded by the EFM have many artifacts due to poor connection along the 
    fiber rotary joint.  We can mask these (see df_fiber_qc), but the artifacts all 
    show up as very high frequency spikes.  It's easier to just filter them out, 
    then we don't have to bother with interpolation later.

    At the same time, we'll pull out the oscillations of interest from the IMU channels, 
    and remove the components we don't want to deal with.

    The time field is going to need to be reconstructed, we don't really want to 
    filter that one if we can avoid it

    adc_offset is the delay in seconds for the adc channel due to the analog ciruit 
    for the charge amplifier on the spheres
    """

    sample_period = np.floor( 1000/sample_rate )
    ###
    # step 1 is to indentify inserted samples, there's a fustratingly large number of these
    # this is done using just the time array
    t = df_fiber['adc_ready_millis']

    print ('Removing inserted junk data')
    m = np.ones( len(df_fiber), dtype='bool')
    for i in range(1,len(df_fiber)-1):
        #insertion errors, we've inserted some junk data
        #we find these by looking at the time gap between 3 samples, 
        #and if it's the gap we expect between 2, then bob's some junk data
        if t[i+1]-t[i-1]>= sample_period and t[i+1]-t[i-1]<=sample_period+1:
            m[i] = False

    #I don't get data frames, they always just get in my way, 
    #getting rid of them for the betterment of mankind
    #for a bit, I'll put them back, I promise
    series = {}
    for arr_name in df_fiber:
        series[ arr_name ] = np.array( df_fiber[arr_name][m] )

    
    # df_fiber = df_fiber[m]

    print( 'indentifying sections that need reconstruction' )
    t = series['adc_ready_millis']
    #reconstruction part, this will be painful
    reconstruction_indices = []
    iStart = 0   #everything is good
    for i in range(1,len(t)-1):
        if t[i] - t[i-1] < sample_period or t[i] - t[i-1] > sample_period+1:
            #that's not ideal, something has shifted
            if t[i+1] - t[i] < sample_period or t[i+1] - t[i] > sample_period+1:
                if iStart == 0:
                    iStart = i

        elif iStart != 0:
            dt = (t[i]-t[iStart-1]) / ( i - iStart )
            if dt <= sample_period+1:
                #we've returned the status quo
                #note, there may have been some crap inserted, which is why the lower bound check is not made
                reconstruction_indices.append( [iStart, i-1] )
                iStart = 0

    #now we need to calculate how many samples need to be removed from each reconstruction
    print( 'removing excess samples from reconstructed regions' )
    for i in range( len(reconstruction_indices) ):
        i0,i1 = reconstruction_indices[i]
        dt = t[i1] - t[i0-1]    #the time difference
        dn = round( dt*sample_rate/1000 -1 ) #sample different
        to_remove = (i1-i0) - dn    #the number of extra samples
        # print( i0, i1-i0, dt*sample_rate/1000, to_remove)
        if to_remove >0:
            m = np.ones( len(t), dtype='bool')
            m[i0:i0+to_remove] = False
            #update all the indices to account for the removed samples
            reconstruction_indices[i][1] -= to_remove
            for j in range( i+1, len(reconstruction_indices) ):
                reconstruction_indices[j][0] -= to_remove
                reconstruction_indices[j][1] -= to_remove
            #remove the samples
            for k in series:
                series[k] = series[k][m]
            #update the time array (may not be required because pointers)
            t = series['adc_ready_millis']
    
    #try to do the reconstruction
    print( 'attempting to reconstruct regions with cosine functions' )
    for i0,i1 in reconstruction_indices:
        
        # reconstruct time
        # this is done linearly
        t = series[ 'adc_ready_millis' ]
        v = t[i0-1]
        i = i0
        while i < i1:
            v += 1000/sample_rate
            t[i] = int(v)
            i += 1

        # reconstruct the IMU fields
        # this is done with a cosine
        for fieldName in ['magnetometer_x', 'magnetometer_y', 'magnetometer_z', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'adc_volts']:
            data = series[fieldName]

            #we need to expand the field a bit to get the entire flat spot
            #it's always a bit bigger than it was in the time field
            j0 = i0
            while data[j0] == data[j0+1] or data[j0] == data[j0-1]:
                j0 -= 1
            j1 = i1+1
            while data[j1] == data[j1+1] or data[j1] == data[j1-1]:
                j1 += 1

            #now we get a snippet of data
            amp = data[j0-100:j0].max()
            fit = optimize.curve_fit( cosFn, t[j0-100:j0], data[j0-100:j0], p0=[amp,2,0,0] )
            
            data[j0:j1] = cosFn( t[j0:j1], *fit[0] )

    #we've now gotten almost all of it
    #there will still be a little bit of cruft left
    #look for outliers in the output, then fill them in using linear interpolation
    print( 'Removing outlier noise' )
    for fieldName in ['magnetometer_x', 'magnetometer_y', 'magnetometer_z', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'adc_volts']:
        #get a threshold
        dv = 10*np.quantile( abs( np.diff( series[fieldName] ) ), .95  )
        for i in range( 1, len( series[fieldName]) ):
            if abs(series[fieldName][i]) > abs(series[fieldName][i-1]) + dv:
                #something is wrong, this is way too big.  Keep going until it isn't
                if i0 is None:
                    i0 = i-1
            elif i0 is not None:
                if not abs(series[fieldName][i]) > abs(series[fieldName][i0]) + dv:
                    print( i0, i )
                    #we had found something, and now we need to do a linear interp
                    v0 = series[fieldName][i0]
                    v1 = series[fieldName][i]

                    #what's the slope?
                    m = (v1-v0)/(i-i0)

                    #then the equation is v = v0 + m*(i-i0)
                    x = np.arange( i0, i )
                    series[fieldName][i0:i] = v0+m*(x-i0)
                    
                    #reset things
                    i0 = None

    #last on the list is to correct the delay in the voltage signal
    v = np.fft.fft( series['adc_volts'] )
    df = np.exp( -2J*np.pi*np.fft.fftfreq(len(v))*sample_rate*adc_offset )
    series['adc_volts'] = np.fft.ifft( v*df ).real    

    #we've removed a bunch of stuff, now we have to deal with the shit pd dataframe
    #why are we using data frames again?
    #seems like the easiest way to deal with this is to just make a whole new frame
    #we can't overwrite the old field names, because reasons
    df = pd.DataFrame()
    for field_name in series:
        #while we're doing this, make sure there's an even number of samples
        #this will make later FFT based filtering less annoying
        if len( series[field_name] ) %2 == 0:
            df[field_name] = series[field_name]
        else:
            df[field_name] = series[field_name][:-1]
    return df

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
    #TODO redo all this to use FFTs, which don't need to deal with any of this mucking about
    from scipy.signal import resample

    t = df.adc_ready_millis*1.0e-3 # seconds
    delta_ts = t.diff()
    fs = 1.0/delta_ts.median()
    new_fs = fs*up

    dn = int(np.floor(new_fs*dt)) # Number of samples to advance
    if dn == 0:
        return df[adc_var].copy()

    n_orig = df[adc_var].shape[0]
    n_up = n_orig*up
    adc_up = resample(df[adc_var], n_up)
    
    #do not shift the data before filtering the data, it's a bad idea
    print("Shifting ADC backward by {0:3.1f} original samples.".format(new_fs*dt/up))
    new_adc = adc_up[dn::up]
    n_pad = n_orig - new_adc.shape[0]
    adc_volts_shifted = pd.Series(
        np.hstack([new_adc, np.zeros(n_pad)])
    )
    assert adc_volts_shifted.shape[0] == n_orig
    return adc_volts_shifted