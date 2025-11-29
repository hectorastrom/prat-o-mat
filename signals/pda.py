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
from scipy.stats import norm
from pprint import pprint


# males are ~80-180Hz (lows at 65); females are 160-280Hz (highs at 1000Hz)
def compute_f0_fingerprint(x : np.ndarray, fs: float, time_range) -> tuple[float, float]:
    """
    Return a normal distribution representation of the fundamental frequency (in
    Hz) of a speaker.
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
    
    mu = float(np.mean(voiced_f0))
    std = float(np.std(voiced_f0))

    return norm(loc=mu, scale=std)

hector_range = [0.66, 1.92]  # s
albert_range = [2.70, 3.65]  # s
x, fs = load("audio/test.wav")

speaker_fingerprints = {
    "hector": compute_f0_fingerprint(x, fs, hector_range), # scipy.stats.norm
    "albert": compute_f0_fingerprint(x, fs, albert_range),
    "none": norm(0, 1) # not relevant what this distribution is
}
speaker_names = speaker_fingerprints.keys()
print(f"Speaker freq ranges: {speaker_fingerprints}")
total_times = {speaker : 0.0 for speaker in speaker_names}

# now the next step is to actually break the signal up by when someone's
# speaking
post_calibration_start = max([*hector_range, *albert_range])
post_calibration_start_idx = int(post_calibration_start * fs)
post_calibration_segment = x[:post_calibration_start_idx]

f0_track, voiced_flag_track, _ = pyin(
    y=post_calibration_segment,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C6'),
    sr=fs,
    frame_length=2048, # default
    hop_length=512 # default
)

time_granularity = float(512 / fs)

def select_dominant_speaker(f0_of_frame : float, is_voiced: bool) -> str:
    """
    Compares fundamental freq (F0) of a frame against the speaker's baselines.
    
    Does so by computing the probability of observing the measured F0 for each
    speaker, and selecting the speaker with the highest probability. This is
    based on the univariate Gaussian modelling the range of F0 for each speaker.
    
    Only works for one speaker at a time!
    """
    if not is_voiced:
        return 'none'
    
    # check belonging to any speaker range
    probs = {speaker: 0.0 for speaker in speaker_names}
    for name, dist in speaker_fingerprints.items():
        # compute pdf of f0 val
        prob = dist.pdf(f0_of_frame)
        probs[name] = prob

    max_likelihood_speaker = max(probs, key=probs.get)
    max_prob = probs[max_likelihood_speaker]
    # final check if there was no near labelled speaker
    if max_prob == 0.0:
        return 'none'

    return max_likelihood_speaker
# Classify each segment according to speaker assignment
print("Starting this heinous loop...")
for f0_of_frame, is_voiced_frame in zip(f0_track, voiced_flag_track):
    cur_speaker = select_dominant_speaker(f0_of_frame, is_voiced_frame)
    total_times[cur_speaker] += time_granularity

pprint(total_times)