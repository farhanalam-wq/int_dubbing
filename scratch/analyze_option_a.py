import json
import math
import struct
import sys
import wave
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Load segments
final_json_path = Path("data/stage_06_translated/clip001__translated__final.json")
with open(final_json_path, "r", encoding="utf-8") as f:
    final_data = json.load(f)

# Segments 002 and 003
seg002 = [s for s in final_data["segments"] if s["segment_id"] == "clip001_seg002"][0]
seg003 = [s for s in final_data["segments"] if s["segment_id"] == "clip001_seg003"][0]

start_ms = seg002["start_ms"]
end_ms = seg003["end_ms"]
total_dur_s = (end_ms - start_ms) / 1000.0

print(f"Option A range: {start_ms}ms to {end_ms}ms ({total_dur_s:.2f}s)")
print("Source Hindi Seg 002:", seg002["source_text"])
print("Target Bangla Seg 002:", seg002.get("translated_text_final"))
print("Source Hindi Seg 003:", seg003["source_text"])
print("Target Bangla Seg 003:", seg003.get("translated_text_final"))

# 2. Check aligned words & silences
aligned_words_path = Path("data/stage_04_aligned/clip001__aligned__words.json")
with open(aligned_words_path, "r", encoding="utf-8") as f:
    aligned_data = json.load(f)
all_words = aligned_data.get("words", [])

segment_words = [w for w in all_words if start_ms <= w.get("start_ms", 0) <= end_ms]

print(f"\nAligned Words in Window ({len(segment_words)} words):")
prev_end = start_ms
prev_word = ""
silences = []
for w in segment_words:
    gap = w["start_ms"] - prev_end
    if gap > 200 and prev_word:
        silences.append((prev_end, w["start_ms"], gap, prev_word, w["word"]))
        print(f"  [PAUSE: {gap}ms between '{prev_word}' and '{w['word']}']")
    print(f"  {w['word']:<15} {w['start_ms']}ms -> {w['end_ms']}ms ({w['end_ms']-w['start_ms']}ms)")
    prev_end = w["end_ms"]
    prev_word = w["word"]

# 3. Analyze Audio Acoustics (RMS Energy, Peak Ratio, Dynamics)
vocal_wav = Path("data/stage_02_separated/clip001__separated__vocal.wav")
with wave.open(str(vocal_wav), "rb") as wf:
    n_ch = wf.getnchannels()
    sr = wf.getframerate()
    sw = wf.getsampwidth()
    
    start_frame = int((start_ms / 1000.0) * sr)
    end_frame = int((end_ms / 1000.0) * sr)
    wf.setpos(start_frame)
    raw = wf.readframes(end_frame - start_frame)

total_samples = len(raw) // (sw * n_ch)
unpacked = struct.unpack(f"<{len(raw)//2}h", raw)

# Downmix to mono float
mono = []
for i in range(0, len(unpacked), n_ch):
    mono.append(sum(unpacked[i:i+n_ch]) / float(n_ch))

# Calculate RMS energy in 50ms windows
win_size = int(sr * 0.05)
energies = []
for i in range(0, len(mono) - win_size, win_size):
    chunk = mono[i:i+win_size]
    rms = math.sqrt(sum(x*x for x in chunk) / len(chunk))
    energies.append(rms)

max_energy = max(energies)
mean_energy = sum(energies) / len(energies)
dynamic_range = max_energy / (mean_energy + 1e-6)

# Opening burst energy (first 0.8s vs remainder)
opening_frames = int(sr * 0.8)
opening_rms = math.sqrt(sum(x*x for x in mono[:opening_frames]) / opening_frames)
rest_rms = math.sqrt(sum(x*x for x in mono[opening_frames:]) / (len(mono) - opening_frames))
opening_boost = opening_rms / (rest_rms + 1e-6)

print(f"\nAcoustic Profile:")
print(f"  Mean RMS: {mean_energy:.1f}, Max RMS: {max_energy:.1f}")
print(f"  Dynamic Range Ratio: {dynamic_range:.2f}x")
print(f"  Opening Burst ('সংঘর্ষ!') Boost Ratio: {opening_boost:.2f}x louder than rest")
