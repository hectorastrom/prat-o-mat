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
from scipy.signal import stft
from pprint import pprint

# males are ~80-180Hz (lows at 65); females are 160-280Hz (highs at 1000Hz)
def compute_pitch(x : np.ndarray, fs: float, time_range) -> float:
    """
    Return min, max, and average pitch of a speaker over a given range.
    """
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

    return float(np.min(voiced_f0)), float(np.max(voiced_f0)), float(np.mean(voiced_f0))

hector_range = [0.64, 1.92]  # s
albert_range = [2.70, 3.65]  # s
x, fs = load("audio/test.wav")

speakers = {
    "hector": compute_pitch(x, fs, hector_range), # min, max, avg
    "albert": compute_pitch(x, fs, albert_range)
}
print(f"Speaker freq ranges: {speakers}")
total_times = {speaker : 0.0 for speaker in speakers.keys()}
total_times['none'] = 0.0 # track non-speaker time


# now the next step is to actually break the signal up by when someone's
# speaking
post_calibration_start = max([*hector_range, *albert_range])
post_calibration_start_idx = int(post_calibration_start * fs)
post_calibration_segment = x[post_calibration_start_idx:]

N, overlap = 512, 256
time_granularity = (N-overlap) / fs 
f, t, Zkk = stft(x[post_calibration_start_idx:], fs, "hann", nperseg=N, noverlap=overlap, return_onesided=True)
assert time_granularity == float(t[1] - t[0]), "your math is wrong"

# now we need to classify each segment as either having a dominant frequency in
# one of the two

def compare_dominant_pitch(segment: np.ndarray) -> str:
    """
    CURRENTLY ONLY SUPPORTS ONE SPEAKER AT TIME
    
    Checks if highest magnitude frequency is in one of the speaker's ranges. If
    so, it returns the name of the speaker. Else, returns 'none'. 
    """
    # note, this could be adjusted to top k; but then we also have to deal with
    # harmonics possibly being second, third, and fourth highest before another
    # speaker...
    top_freq_idx = np.argmax(np.abs(segment))
    top_freq = float(f[top_freq_idx])
    # check if top_freq in range of any speaker range
    distances = {speaker : fs // 2 + 1 for speaker in speakers.keys()} # max f distance
    for name, (lo, hi, avg) in speakers.items():
        if lo <= top_freq <= hi:
            distances[name] = abs(top_freq-avg) # distance from average
            
    min_speaker = min(distances, key=distances.get)
    min_value = distances[min_speaker]
    if min_value == fs // 2 + 1:
        return 'none'
    return min_speaker

# I know, this iteration is terribly inefficient. Let's think about how to do
# this smarter later
# Zkk : (F, T)
print("Starting this heinous loop...")
for segment in Zkk.T: # iterate through times
    # segment: (F,)
    cur_speaker = compare_dominant_pitch(segment)
    total_times[cur_speaker] += time_granularity

pprint(total_times)

# THIS STFT APPROACH IS FLAWED
# I'M COMMITTING IT BECAUSE IT'S PART OF THE PROCESS
# The problem is that the frequency with the highest magnitude is NOT the
# fundamental frequency. Aside from this approach being inefficient, it's also
# incorrect. The highest amplitude frequency from speech often turns out to be
# the 2nd or 3rd harmonic, which will be out of the range of the F0's we
# identified for each subject.

# To correct for this, we can just use what's already computed with PYIN. PYIN
# runs along the full duration of the signal, breaking each segment (length 256
# samples) into a label of its f0 and whether or not that segment is considered
# 'voiced'. To then check speaker ownership, we can iterate through each segment
# and assign that segment to the speaker with the min abs distance from f0 of
# the segment. 

# This is still not great, but will be the subject of my next commit
