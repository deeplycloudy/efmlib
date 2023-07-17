import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, sys
# from scipy.signal import medfilt, find_peaks
from scipy import optimize

import efmlib

fftLength     = 512   #samples
fftInterp     = 10     #factor
stepLength    = 45     #samples
sampleRate    = 45     #Hz
spinRate      = [1,3]  #spin rate should be in the range, Hz
spinThreshold = 2000   #checks that the accellerometer indicates the spheres are actually spinning
# gpsTimeOffset = 71300  #IOP2-sleet in millis, this number accounts for GPS and adc_volts not being sync'd.  Not the same for all flights
# gpsTimeOffset = 167477  #IOP2-graupel in millis, this number accounts for GPS and adc_volts not being sync'd.  May not be the same for all flights
# gpsTimeOffset = 167477  #IOP2-graupel in millis, this number accounts for GPS and adc_volts not being sync'd.  May not be the same for all flights

# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP2/Sleet/EFM[01]*TXT' ) )

# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP2/Sleet-2007-11-2//EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP2/Graupel-2007-38-1/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP4/Blizzard/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP4/Thunder-2007-15-2/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP5/SNOW-2007-25-1/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP6/RUST-IOP6-2007-40-1/EFM[012]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP8/IOP8_2007-04-07_frost/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP8/IOP8_2007-36-11_Ice/EFM[01]*TXT' ) ) #altitude information for this flight looks sus
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP10/IOP10_2007-11-2_sleet/EFM[012]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP10/IOP10_2007-15_2_thunder/EFM[01]*TXT' ) )
filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP11/EFM*TXT' ) )


df_gps, df_fiber = efmlib.io.read_efm_raw(filenames, shift_dt=0.0)
#filter the data
df_fiber = efmlib.qc.df_fiber_filter( df_fiber, adc_offset=0.035 )


###
# get the offset betweel GPS milli and fiber_millis
# these do not start at the same place, which is annoying.  We'll do this with the alitude and pressure signals
# since they both show launch quite clearly
#the ground pressure and altitude are very approximate.  It's to identify yad at the beginning of the flight
pGround = 1000
zGround = 100     
#the thresholds are used to detect launch, these are less approxiamte
pThresh = 3       #mb, launch detect
zThresh = 20      #m, launch detect
#find ground
iFiber = 0
while abs(df_fiber['pressure'][iFiber] - pGround) > 50:
    iFiber += 1
pGround = df_fiber['pressure'][iFiber]

iGps = 0
while abs(df_gps['altitude'][iGps] - zGround) > 150:
    iGps += 1
zGround = df_gps['altitude'][iGps]

while pGround - df_fiber['pressure'][iFiber] < pThresh:
    iFiber += 1
while df_gps['altitude'][iGps] - zGround < zThresh:
    iGps += 1
gpsMilliOffset = df_fiber['adc_ready_millis'][iFiber] - df_gps['millis'][iGps]
print( gpsMilliOffset )


iSample = fftLength//2
freq = np.fft.rfftfreq( fftLength*fftInterp )*sampleRate
m    = ( freq>spinRate[0] )&( freq<spinRate[1] )
Ay = df_fiber['acceleration_y']
V  = df_fiber['adc_volts']
E = []
t = []
while iSample < len( df_fiber ) -fftLength//2:
    #get the fft for Ay
    AyT = Ay[ iSample-fftLength//2:iSample+fftLength//2 ]
    AyF = np.fft.rfft( AyT-AyT.mean(),  fftLength*fftInterp) / (fftLength/2)

    VT = V[ iSample-fftLength//2:iSample+fftLength//2 ]
    VF = np.fft.rfft( VT-VT.mean(),  fftLength*fftInterp) / (fftLength/2)

    iMax = abs(AyF[m]).argmax()
    jMax = abs(VF[m]).argmax()
    iGps = abs(df_gps['millis']-df_fiber['adc_ready_millis'][iSample]+gpsMilliOffset).argmin()

    #get the z value
    z = df_gps['altitude'][iGps]
    #check that we're going up
    if len( t ) > 0:
        if t[-1][1] > z:
            #we're not
            iSample += stepLength
            continue

    #are we spinning?
    if abs(AyF[m][iMax]) < spinThreshold:
        #no, we are not
        E.append( [0, 0, 0 ] )
        t.append( [ df_fiber['adc_ready_millis'][iSample] , z, df_fiber['pressure'][iSample]] )
        iSample += stepLength
        continue

    #iMax and jMax should be close, because the voltage is supposed to be oscillating 
    #with the spinning.  If it's not, well lets just 0 it out for now
    if abs( iMax-jMax ) > 5:
        #the magnetude of E is just the mag of VF
        Emag = abs(VF[m])[iMax]
        #we also need to know the phase of the signal though
        Vp  = np.arctan2(  VF[m][iMax].imag,  VF[m][iMax].real )
        Ayp = np.arctan2( AyF[m][iMax].imag, AyF[m][iMax].real )
        # print ('Accellaration and voltage arent synced', df_fiber['adc_ready_millis'][iSample], (iMax-jMax) )

    else:
        #the magnetude of E is just the mag of VF
        Emag = abs(VF[m])[jMax]
        #we also need to know the phase of the signal though
        Vp  = np.arctan2(  VF[m][jMax].imag,  VF[m][jMax].real )
        Ayp = np.arctan2( AyF[m][iMax].imag, AyF[m][iMax].real )
        
    phase = (Vp-Ayp)%(2*np.pi)
    #are we in phase, out out of phase
    #-pi/2 to pi/2 is in phase
    #pi/2 to 3pi/2 is out of phase
    if phase > np.pi/2 and phase < 3*np.pi/2:
        Emag *= -1 


    E.append( [Emag, phase, freq[m][iMax]] )
    t.append( [ df_fiber['adc_ready_millis'][iSample] , z, df_fiber['pressure'][iSample] ] )

    iSample += stepLength

t = np.array( t )
E = np.array( E )

plt.figure()
plt.plot(E[:,0], t[:,1] )
# plt.plot( (E[:,1]-np.pi)/np.pi, t[:,1], '.' )    #this is phase
# plt.plot( E[:,2], t[:,1] )                #this is spin frequency
plt.xlabel( 'Electric field [uncalibrated]' )
plt.ylabel( 'Altitude [m]' )

###
# this is a diagnostic.  If things are working correctly, they should start going up at the same time
plt.figure()
plt.plot( t[:,0], t[:,1] )
plt.plot( t[:,0], 10*(1000-t[:,2]) )
plt.xlabel( 'Time [millis]' )
plt.ylabel( 'Altitude-ish [m]' )

plt.show()