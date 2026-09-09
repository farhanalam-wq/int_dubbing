import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("data/stage_06_translated/clip001__translated__final.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print(f"Total segments: {len(d['segments'])}")
for s in d["segments"]:
    dur = (s["end_ms"] - s["start_ms"]) / 1000.0
    txt = s.get("translated_text_final") or s.get("translated_text_raw")
    print(f"{s['segment_id']}: [{s['start_ms']}ms -> {s['end_ms']}ms] ({dur:.2f}s) | {s['speaker_id']} | {txt}")
