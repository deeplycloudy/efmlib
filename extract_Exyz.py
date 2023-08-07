import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, sys
# from scipy.signal import medfilt, find_peaks
from scipy import optimize

import efmlib

"""
This is a demodulation method based on the Paul Krehbiel's notes of the previous EFM package
To demodulate the vector electric field, they use standard mixing techniques, so you 
multiply by the carrier frequency and average.  In simulations, this worked pretty well, but 
in the real data the demodulated signal is prone to oscillations.  
"""

samplerate    = 45.45   #Hz
spin          = 1.6     #Hz (approximate)
rotation      = 0.7     #Hz (approximate)
spinRate      = [1,3]  #spin rate should be in the range, Hz
spinThreshold = 12000   #checks that the accellerometer indicates the spheres are actually spinning
rotaThreshold = 3000   #checks that the magnetometer indicates the EFM is rotating
signalDelay   = -0.057
###
# these set how much data we use to get phase information for rotation and spin
NSpec = 512
NPadd = 2048
NFit  = 64

filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP2/Sleet/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP11/EFM*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP2/Sleet-2007-11-2//EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP2/Graupel-2007-38-1/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP4/Blizzard/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP4/Thunder-2007-15-2/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP5/SNOW-2007-25-1/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP6/RUST-IOP6-2007-40-1/EFM[012]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP8/IOP8_2007-04-07_frost/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP8/IOP8_2007-36-11_Ice/EFM[01]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP10/IOP10_2007-11-2_sleet/EFM[012]*TXT' ) )
# filenames = sorted( glob.glob( '/localdata/mstock/LEE/IOP10/IOP10_2007-15_2_thunder/EFM[01]*TXT' ) )

# filenames = sorted( glob.glob( '/data/LEE/IOP2/Sleet-2007-11-2//EFM[01]*TXT' ) )
titleStr  = 'IOP2 Sleet'



def cosFn ( t, amplitude, frequency, phase, offset ):
    #force some constraints
    amplitude = abs(amplitude)
    frequency = abs(frequency)

    return  amplitude*np.cos( 2*np.pi*t*frequency + phase ) + offset

def quadmax( arr, iMax=None ):
    if iMax is None:
        iMax = abs(arr).argmax() 
    
    #edge case
    if iMax == 0 or iMax == len(arr)-1:
        return iMax
    
    t = [-1,0,1] 
    p = np.polyfit( t, abs( arr[iMax-1:iMax+2] ), 2 )
    #the exact maxima shows up here
    return iMax -p[1]/p[0]/2


df_gps, df_fiber = efmlib.io.read_efm_raw(filenames, shift_dt=0.0)
#filter the data
df_fiber = efmlib.qc.df_fiber_filter( df_fiber,adc_offset=signalDelay )

Ax = np.array( df_fiber['acceleration_x'] ) #arb units
Bz = np.array( df_fiber['magnetometer_z'] ) #arb units
V  = np.array( df_fiber['adc_volts'] )  #in volts
t  = np.array( df_fiber['adc_ready_millis']/1000 ) #in seconds

################33
# Spin phase
print ('Demodulating Spin' )
f = np.fft.rfftfreq( NPadd ) * samplerate
spinPhase = [0]
spinFreq  = [0]
spinTime  = []

for i in range( NSpec//2, len(t)-NSpec//2 ):
    spinTime.append( t[i] )
    data = Ax[i-NSpec//2:i+NSpec//2]
    if data.max()-data.min() < spinThreshold:
        spinFreq.append(0)
        spinPhase.append(0)
        continue

    v = np.fft.rfft( data, NPadd )
    if spinFreq[-1] != 0:
        #get a mask around where we think the peak should be
        iMax = ( abs(v)*np.exp( -(f-spinFreq[-1])**2 / .5**2 ) ).argmax()
    else:
        iMax = ( abs(v)*np.exp( -(f-spin)**2 / .5**2 ) ).argmax()

    #get a subsample location
    iMax = quadmax( v, iMax )
    #convert into frequency
    spinFreq.append( iMax/NPadd*samplerate )
    if spinFreq[-1] <= 0:
        spinPhase.append(0)
        continue

    NFit = int( 2.0/spinFreq[-1]*samplerate )
    if NFit > NSpec//2: NFit = NSpec//2
    fitTime = np.arange( -NFit/samplerate, NFit/samplerate, 1./samplerate )[:2*NFit]
    data = Ax[i-NFit:i+NFit]
    if data.max()-data.min() < spinThreshold:
        spinPhase.append( 0 )
    else:
        amp    = ( data.max()-data.min() )/2
        offset = data.mean()
        frequency = spinFreq[-1]
        try:
            # fit = optimize.curve_fit( cosFn, fitTime, data , p0=[spinThreshold/2,spinFreq[-1],0,0], bounds=( [0,spinFreq[-1]-0.0001,-np.pi,0], [np.inf, spinFreq[-1]+0.0001,np.pi,np.inf] ) )
            fit = optimize.curve_fit( cosFn, fitTime, data , p0=[amp,frequency,0,offset], bounds=( [.75*amp,frequency-0.01,-np.pi,offset-spinThreshold/10], [1.5*amp,frequency+0.01,np.pi,offset+spinThreshold/10] ) )
            spinPhase.append( fit[0][2] )
        except RuntimeError:
            print( 'Error getting phase fit, %i, %f'%(i, spinFreq[-1]) )
            spinPhase.append( 0 )
            if i > 3860: sys.exit()
    
    if i > 40000: break

spinPhase = np.array( spinPhase[1:] )

print ('Demodulating Rotation' )
rotaPhase = [0]
rotaFreq  = [0]
rotaTime  = []
for i in range( NSpec//2, len(t)-NSpec//2 ):
    rotaTime.append( t[i] )
    data = Bz[i-NSpec//2:i+NSpec//2]
    if data.max()-data.min() < 1.5:
        rotaFreq.append(0)
        rotaPhase.append(0)
        continue

    v = np.fft.rfft( data, NPadd )
    if rotaFreq[-1] != 0:
        #get a mask around where we think the peak should be
        iMax = ( abs(v)*np.exp( -(f-rotaFreq[-1])**2 / .5**2 ) ).argmax()
    else:
        iMax = ( abs(v)*np.exp( -(f-rotation)**2 / .5**2 ) ).argmax()

    #get a subsample location
    iMax = quadmax( v, iMax )
    #convert into frequency
    rotaFreq.append( iMax/NPadd*samplerate )
    if rotaFreq[-1] <= 0:
        rotaPhase.append(0)
        continue

    NFit = int( .75/rotaFreq[-1]*samplerate )
    if NFit > NSpec//2: NFit = NSpec//2
    fitTime = np.arange( -NFit/samplerate, NFit/samplerate, 1./samplerate )[:2*NFit]
    data = Bz[i-NFit:i+NFit]
    if data.max()-data.min() < 1.5:
        rotaPhase.append( 0 )
    else:
        amp    = ( data.max()-data.min() )/2
        offset = data.mean()
        frequency = rotaFreq[-1]
        try:
            # fit = optimize.curve_fit( cosFn, fitTime, data , p0=[spinThreshold/2,spinFreq[-1],0,0], bounds=( [0,spinFreq[-1]-0.0001,-np.pi,0], [np.inf, spinFreq[-1]+0.0001,np.pi,np.inf] ) )
            fit = optimize.curve_fit( cosFn, fitTime, data , p0=[amp,frequency,0,offset], bounds=( [.75*amp,frequency-0.01,-np.pi,offset-spinThreshold/10], [1.5*amp,frequency+0.01,np.pi,offset+spinThreshold/10] ) )
            #since Bz is based on sin(pRot), we need a pi/2 phaseshift here
            p = fit[0][2] + np.pi/2
            if p > np.pi: p-=2*np.pi
            rotaPhase.append( p )
        except RuntimeError:
            print( 'Error getting phase fit, %i, %f'%(i, spinFreq[-1]) )
            rotaPhase.append( 0 )
            if i > 3860: sys.exit()
    

rotaPhase = np.array( rotaPhase[1:] )

#use demodulated phase to get Ez
#this is the associated signal
Vsig = V[NSpec//2:NSpec//2+len(spinPhase) ]

#we need to hpf this stuff, to remove anything lower than about 1Hz so that the demodulation has a place to demodulate into
v = np.fft.rfft( Vsig )
f = np.fft.rfftfreq( len(Vsig)) *samplerate
hpf = 1
filt = np.zeros( len(f) )
filt[ f<hpf ] = np.cos( np.pi*f[ f<hpf ]/hpf/2 )
Vsig = np.fft.irfft( (1-filt)*v )

#This should shift the oscillation stuff up a bunch
Ez_ = np.sin( spinPhase )*Vsig

#to get the DC bit, we filter using an LPF
v = np.fft.rfft( Ez_ )
f = np.fft.rfftfreq( len(Vsig)) *samplerate
lpf = 0.4
filt = np.zeros( len(f) )
filt[ f<lpf ] = .5 + .5*np.cos( np.pi*f[ f<lpf ]/lpf )
#the 2* is expected from the offset
Ez_ = 2* np.fft.irfft( v*filt )


#extract Eh
Eh_ = np.cos( spinPhase )*Vsig
v = np.fft.rfft( Eh_ )
f = np.fft.rfftfreq( len(Eh_)) *samplerate
filt = np.zeros( len(f) )

filt[ f<2*rotation ] = np.sin( np.pi*f[ f<2*rotation ]/2/rotation )
Eh_ = 2*np.fft.irfft( v*filt )
#this still has a rotational component to it
Ey_ =  Eh_*np.sin( rotaPhase )
Ex_ = -Eh_*np.cos( rotaPhase )
f = np.fft.rfftfreq( len(Ez_)) *samplerate
filt = np.zeros( len(f) )
filt[ f<lpf ] = .5 + .5*np.cos( np.pi*f[ f<lpf ]/lpf )

v = np.fft.rfft( Ex_ )
Ex_ = 2* np.fft.irfft( v*filt )
v = np.fft.rfft( Ey_ )
Ey_ = 2* np.fft.irfft( v*filt )



print( '%0.3f %7.4f'%(delay, 1000*noise.std()) )

# bestDelay = 0, 50
# for delay in np.arange( -.1, 0, .001 ):

#     Vsig = V[NSpec//2:NSpec//2+len(spinPhase) ]
#     #delay Vsig
#     v = np.fft.fft( Vsig )
#     N = len(Vsig)
#     df = np.exp( -2J*np.pi*np.fft.fftfreq(N)*(-delay*samplerate) )
#     Vsig = np.fft.ifft( v*df ).real
#     #we need to hpf this stuff, to remove anything lower than about 1Hz so that the demodulation has a place to demodulate into
#     v = np.fft.rfft( Vsig )
#     f = np.fft.rfftfreq( len(Vsig)) *samplerate
#     lpf = 0.2
#     filt = np.zeros( len(f) )
#     filt[ f<lpf ] = np.cos( np.pi*f[ f<lpf ]/lpf/2 )
#     Vsig = np.fft.irfft( (1-filt)*v )

#     #This should shift the oscillation stuff up a bunch
#     Ez_ = np.sin( spinPhase )*Vsig


#     #to get the DC bit, we filter
#     v = np.fft.rfft( Ez_ )
#     #apply filter, and we're 'done'  
#     #the 2* is expected from the offset
#     Ez_ = 2* np.fft.irfft( v*filt )

#     #now HPF it so I can see the oscillation and find the delay that minimizes it
#     filt = np.ones( len(f) )
#     hpf = 0.02
#     filt[ f<hpf ] = 1-np.cos( np.pi*f[ f<hpf ]/hpf/2 )
#     noise  = np.fft.irfft( np.fft.rfft(Ez_)*filt )

#     # print( '%0.3f %7.4f'%(delay, 1000*noise.std()) )
#     if 1000*noise.std() < bestDelay[1]:
#         bestDelay = delay, 1000*noise.std()
# print( '%0.3f %7.4f'%bestDelay )

# plt.plot( noise )