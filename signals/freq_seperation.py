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
t_resolution = nperseg / fs
f_resolution = 1/t_resolution
print(f"Segments are {t_resolution:.2f}s long, with {f_resolution} frequency granularity")
# f is array of all represented frequencies: f[0] is 0Hz, f[-1] is fs/2 Hz
f, t, Zkk = stft(data, fs, 'hann', nperseg=nperseg, noverlap=256, return_onesided=True)
print(f[:5])

# Test
# convert k's to frequencies
desired_time_s = 3 # what exact s you want to look at
s_idx = int(desired_time_s / t_resolution)
freq_mags_at_t = np.abs(Zkk[:, s_idx])
highest_freq_idx = np.argmax(freq_mags_at_t)
highest_freq = float(f[highest_freq_idx])
print(f"{highest_freq=:.2f}Hz")

###############################
# Extract intro stage bands
###############################
hector_range = [0.64, 1.92] # s
albert_range = [2.70, 3.65] # s

def average_freq_over_range(range, x, fs, nperseg=1024):
    """
    Compute the (avg, min, max) frequencies over a range of times
    
    Inputs:
        x (np.ndarray) - raw signal
        fs (float) - sampling rate
    """
    assert len(range) == 2, "improper range"
    lo, hi = range
    assert lo < hi, "lo > hi?"

    f, t, Zkk = stft(
        x, fs, "hann", nperseg=nperseg, noverlap=256, return_onesided=True
    )
    f_granularity = float(f[1] - f[0])
    t_granularity = float(t[1] - t[0])

    seen_min, seen_max = float('inf'), float('-inf')
    avg_over_time = []

    lo_idx = int(lo / t_granularity)
    hi_idx = int(hi / t_granularity)
    t_idx = np.arange(lo_idx, hi_idx)
    for idx in t_idx:
        freq_mags = np.abs(Zkk[:, idx]) # magnitude of freqs at this timestep

        # if we normalize so max freq is 1 and min freq is 0, then we multipy by
        # the mags by the corresponding freq values, then we get a weighted
        # average of the frequencies. just have to divide again by the sum of
        # the normalized vector
        min_mag, max_mag = freq_mags.min(), freq_mags.max()
        normalized_freq_mags = (freq_mags - min_mag) / (max_mag - min_mag)
        weighted_freqs = normalized_freq_mags * f # (f,) * (f,)
        avg_freq = np.mean(weighted_freqs)
        avg_over_time.append(float(avg_freq))

    return np.array(avg_over_time)

hector_avgs = average_freq_over_range(hector_range, data, fs)   
print(hector_avgs)     
print(f"hector's overall average freq: {float(np.mean(hector_avgs))}")

albert_avgs = average_freq_over_range(albert_range, data, fs)
print(albert_avgs)
print(f"albert's overall average freq: {float(np.mean(albert_avgs))}")
