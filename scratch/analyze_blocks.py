import os
import sys
import wave
import struct
import math
from pathlib import Path

output_dir = Path("data/stage_07_tts")

print("=" * 80)
print("BLOCK ACOUSTIC INSPECTION")
print("=" * 80)

for f in sorted(os.listdir(output_dir)):
    if not (f.startswith("block_") and f.endswith(".wav")):
        continue
    filepath = output_dir / f
    with wave.open(str(filepath), "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        dur = n_frames / float(sr)
        raw = wf.readframes(n_frames)
        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        max_amp = max(abs(x) for x in samples)
        rms = math.sqrt(sum(x*x for x in samples) / len(samples))
        
        # Check non-silent duration (frames with abs(sample) > 500)
        speech_samples = [i for i, x in enumerate(samples) if abs(x) > 500]
        if speech_samples:
            speech_start = speech_samples[0] / float(sr)
            speech_end = speech_samples[-1] / float(sr)
            actual_speech_dur = speech_end - speech_start
        else:
            speech_start, speech_end, actual_speech_dur = 0, 0, 0
            
        print(f"{f:<35} | Dur: {dur:>5.2f}s | Speech: {speech_start:>4.2f}s -> {speech_end:>4.2f}s ({actual_speech_dur:>5.2f}s) | Max: {max_amp:>5} | RMS: {rms:>5.1f}")
