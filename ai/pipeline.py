import torch
import torchaudio
import pandas as pd
from pyannote.audio import Pipeline
import os
from pyannote.audio.core import task as pyannote_task

HF_TOKEN = os.getenv("HF_TOKEN")
AUDIO_FILE = "audio/test.wav"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

safe_types = [
    pyannote_task.Specifications,
    pyannote_task.Problem,
    pyannote_task.Resolution,
    torch.torch_version.TorchVersion,
]

torch.serialization.add_safe_globals(safe_types)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

pipeline.to(device)

print(f"Processing {AUDIO_FILE}...")
waveform, sample_rate = torchaudio.load(AUDIO_FILE)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})


annotation = diarization.speaker_diarization
data = []

for segment, track, label in annotation.itertracks(yield_label=True):
    data.append({
        "start": segment.start,
        "end": segment.end,
        "duration": segment.end - segment.start,
        "speaker": label
    })


df = pd.DataFrame(data)


print(f"Total Speakers Detected: {df['speaker'].nunique()}")
print("\nTotal Speaking Time per Speaker (seconds):")
print(df.groupby("speaker")["duration"].sum())

print("\nSegment Data:")
print(df.head())



