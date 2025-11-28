import soundfile as sf
import numpy as np

data, fs = sf.read("audio/test.wav")
print(fs)