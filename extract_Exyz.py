import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, sys
# from scipy.signal import medfilt, find_peaks
# from scipy import signal

import efmlib

"""
This is a demodulation method based on the Paul Krehbiel's notes of the previous EFM package
To demodulate the vector electric field, they use standard mixing techniques, so you 
multiply by the carrier frequency and average.  In simulations, this worked pretty well, but 
in the real data the demodulated signal is prone to oscillations.  This seems to be related 
to pendulum procession and tilt
"""

filenames = sorted( sys.argv[1:] )


df_gps, df_fiber = efmlib.io.read_efm_raw(filenames, shift_dt=0.0)
#filter the data
df_fiber = efmlib.qc.df_fiber_filter( df_fiber )

At = np.array( df_fiber['acceleration_x'] ) #arb units - tangential direction
Ba = np.array( df_fiber['magnetometer_z'] ) #arb units - axial direction
V  = np.array( df_fiber['adc_volts'] )  #in volts
t  = np.array( df_fiber['adc_ready_millis']/1000 ) #in seconds

#this demodulates the signals in to the x,y,z components.  Units will still be in volts though
Vx, Vy, Vz = efmlib.signal.demodulate_voltage( V, At, Ba )

#This converts volts into V/m electric field
Ex = efmlib.calibration.calibrated_E( Vx )
Ey = efmlib.calibration.calibrated_E( Vy )
Ez = efmlib.calibration.calibrated_E( Vz )
