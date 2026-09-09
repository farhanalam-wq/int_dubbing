import wave
import struct
import math

with wave.open("data/stage_07_tts/block_05_dilemma__tts__cloned.wav", "rb") as wf:
    sr = wf.getframerate()
    n = wf.getnframes()
    raw = wf.readframes(n)
    samples = struct.unpack(f"<{len(raw)//2}h", raw)

# Compute RMS in 1-second chunks
chunk_size = sr
print("Block 5 second-by-second RMS:")
for sec in range(0, len(samples), chunk_size):
    chunk = samples[sec:sec+chunk_size]
    rms = math.sqrt(sum(x*x for x in chunk)/len(chunk))
    print(f"Sec {sec//sr:2d} - {sec//sr + 1:2d}: RMS = {rms:6.1f} | Max = {max(abs(x) for x in chunk):5d}")
