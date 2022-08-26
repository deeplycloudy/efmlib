import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
from scipy.signal import medfilt
# import glob
import sys

import efmlib

plt.rcParams['figure.figsize'] =  10, 7.5
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['text.color'] = 'black'
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.minor.size'] = 5

EFMlist = sys.argv[1:]
EFMlist.sort()

df_gps, df_fiber = efmlib.io.read_efm_raw(EFMlist)
df_fiber = efmlib.qc.df_fiber_qc(df_fiber)

fields = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'magnetometer_x', 'magnetometer_y', 'magnetometer_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']

fig, axes = plt.subplots(figsize=(15, 25), nrows=len(fields), sharex=True)
for fname, ax in zip(fields, axes):
    ax.plot(medfilt(df_fiber[fname],3), color='k')
    ax.set_ylabel(fname)
#ax.set_xlim(4000, 4200)
plt.savefig('rotation.png')
plt.close()

# df_fiber.adc_reading.plot()
# plt.ylim(-1e3,1e3)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df_fiber.adc_ready_millis/1000, medfilt(df_fiber['adc_volts'],3), color='k')
ax.set_ylabel('ADC [Volts]')
#ax.set_xlim(4500,8000)
ax.set_ylim(-.1e-3,.15e-3)
plt.savefig('voltage.png')
plt.close()