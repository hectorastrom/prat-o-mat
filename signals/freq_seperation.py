# @Time    : 2025-11-28 17:01
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : freq_seperation.py
"""
Seperate the conversational members by their assigned frequencies
1. STFT the signal 

Later:
- Determine which part of the audio is the calibration (people announcing their
names) and which is the 'action'
- Continually update frequency bands for members as they're speaking
- 
"""
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import stft


###############################
# Reading file
###############################
fs, data = read("audio/test.wav")
print(fs)
print(data)


###############################
# STFT
###############################
duration = len(data) / fs
N = duration * fs

nperseg = 1028
t_granularity = nperseg / fs
print(f"Segments are {t_granularity:.2f}s long - that's our granularity")
f, t, Zkk = stft(data, fs, 'hann', nperseg=nperseg, noverlap=256)
print(f[:5])

# convert k's to frequencies
desired_time_s = 3 # what exact s you want to look at
s_idx = int(desired_time_s / t_granularity)
freq_mags_at_t = np.abs(Zkk[:, s_idx])
highest_freq_idx = np.argmax(freq_mags_at_t)
highest_freq = float(f[highest_freq_idx])
print(f"{highest_freq=:.2f}Hz")

