# @Time    : 2025-11-28 17:57
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : pda.py
"""
The previous method was (intentionally) naive. There's a whole field of 'pitch
detection algorithms' (PDAs) to determine a speaker.

In this file, we'll be implementing one of them, known as PYIN, built to
estimate the fundamental frequency of a signal (which is the inverse of the
period of a signal).

Roughly, the standard YIN algorithm measures how much a
signal differs from a shifted version of the signal at all possible shifts --
the minimum difference gives the periodicity. The PYIN algorithm extends this
with Viterbi Decoding, which computes the probability of each pitch by looking
at how raplidly it changes in the signal and how likely it is to be voiced (i.e.
not background noise).

See these for more information:
- https://librosa.org/doc/main/generated/librosa.pyin.html
- https://www.eecs.qmul.ac.uk/~simond/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf 
"""
import librosa
from librosa import pyin, load
import numpy as np

# males are ~80-180Hz (lows at 65); females are 160-280Hz (highs at 1000Hz)
def compute_avg_pitch(x : np.ndarray, fs: float, time_range) -> float:
    lo, hi = time_range
    start_sample = int(lo * fs)
    end_sample = int(hi * fs)
    segment = x[start_sample:end_sample]
    
    # using PYIN
    # f0 contains fundamental frequency @ each frame (256 samples)
    f0, voiced_flag, voiced_probs = pyin(
        y=segment,
        fmin=librosa.note_to_hz('C2'), # lower bound of 65Hz
        fmax=librosa.note_to_hz('C6'),
        sr=fs
    )
    
    # select only fundamental frequencies that come from voiced segments
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return 0.0 # no voice detected
    
    return float(np.mean(voiced_f0))

hector_range = [0.64, 1.92]  # s
albert_range = [2.70, 3.65]  # s
data, fs = load("audio/test.wav")

hector_avg_pitch = compute_avg_pitch(data, fs, hector_range)
albert_avg_pitch = compute_avg_pitch(data, fs, albert_range)


print(hector_avg_pitch)
print(albert_avg_pitch)
