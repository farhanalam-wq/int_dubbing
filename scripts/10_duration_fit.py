#!/usr/bin/env python3
"""
Stage 10 — Duration Fit
========================
Reads:  data/stage_07_tts/{clip_id}_{seg_id}__tts__{model}.wav   ← TTS output per segment
        data/stage_05_timeline/{clip_id}__timeline.json            ← source segment durations
Writes: data/stage_08_duration_fit/{clip_id}_{seg_id}__fit__{model}.wav

Tolerance rules (§4.3):
  - If TTS output is within ±150ms of source segment duration → copy as-is
  - If beyond tolerance → apply FFmpeg atempo correction (0.85x–1.15x range)
  - If required stretch exceeds 0.85x–1.15x bounds → flag needs_review, copy as-is
    (do NOT force a bad stretch)

Runs on whichever TTS model is nominated as the bake-off winner (or all, during bake-off).

Usage:
  python scripts/10_duration_fit.py --clip-id clip001 --model dhvaani
"""

# TODO: Implement
# 1. For each segment:
#    a. Get source duration from timeline (end_ms - start_ms)
#    b. Get TTS output duration via ffprobe
#    c. Compute delta; if within tolerance → copy
#    d. Compute required atempo rate; clamp to [0.85, 1.15]
#    e. If clamped rate would not achieve target → flag needs_review, copy
#    f. Else → apply ffmpeg atempo filter
# 2. Update duration_fit_ms in timeline JSON

raise NotImplementedError("Stage 10 not yet implemented. See docstring for plan.")
