# TASK.md — Hindi→Bangla Mythological Dubbing Pipeline

This document is the sole source of truth for building and running this pipeline.
Do not ask the user for clarification on anything covered here — if a decision
point isn't covered, make the most defensible engineering call, note it in
`DECISIONS.md` (create if absent), and proceed.

---

## 0. Project Goal

Take a Hindu mythological video clip (Hindi dialogue), and produce a
lip-synced, dubbed output in Bangla, while:
- Preserving background score/BGM from the original.
- Preserving per-character voice identity and the *emotional/tonal performance*
  of the original delivery (not flat/robotic dubbed audio).
- Correctly handling tatsam (Sanskrit-loanword-heavy) vocabulary in the source
  dialogue — this is a known failure point for generic ASR/MT/TTS and must be
  explicitly validated, not assumed to work.

Current phase: **experimentation on individual clips**, not batch/episode
processing. Build the pipeline so it works stage-by-stage on one clip first,
with every intermediate artifact inspectable, before any batching or
parallelization work happens.

Scope lock: Hindi → Bangla only for this phase. Do not generalize to other
language pairs yet — hardcode Hindi/Bangla in configs is acceptable; note
where a hardcode exists so it's easy to parameterize later.

---

## 1. Repository / Directory Structure

```
project-root/
├── TASK.md                      # this file
├── DECISIONS.md                 # log of engineering calls made without user confirmation
├── models/                      # downloaded model weights/checkpoints (gitignored)
├── configs/
│   └── pipeline.yaml            # model paths, thresholds, glossary path
├── data/
│   ├── raw/                     # input clips, untouched
│   ├── stage_01_extracted/      # ffmpeg-extracted audio (wav, 16kHz mono for ASR; also keep 48kHz stereo master)
│   ├── stage_02_separated/      # vocal.wav + bgm.wav per clip
│   ├── stage_03_transcript/     # raw ASR output (per-engine) + merged
│   ├── stage_04_aligned/        # word-timestamped + diarized transcript
│   ├── stage_05_timeline/       # canonical JSON timeline (see §5 schema)
│   ├── stage_06_translated/     # Bangla translation, pre- and post-glossary-cleanup
│   ├── stage_07_tts/            # per-segment dubbed Bangla audio, per-candidate-model
│   ├── stage_08_duration_fit/   # duration-corrected dubbed segments
│   ├── stage_09_remix/          # dubbed vocal + original BGM remixed track
│   ├── stage_10_lipsync/        # final lip-synced video
│   └── final/                   # muxed final deliverable
├── glossary/
│   └── mythology_glossary.json  # proper nouns, deity names, epithets: Hindi -> Bangla fixed mappings
├── scripts/                     # one script per stage, numbered to match data/ folders
└── logs/
```

Every stage script reads from its numbered input folder and writes to its
numbered output folder. Never overwrite an earlier stage's folder. This keeps
every intermediate artifact inspectable for debugging without rerunning
upstream stages.

File naming convention across all stages: `{clip_id}__{stage_name}__{variant}.{ext}`
e.g. `clip001__tts__dhvaani.wav`, `clip001__tts__fishspeech.wav`,
`clip001__stt__indicconformer.json`. The `{variant}` suffix is mandatory
whenever a stage has a primary/fallback/bake-off choice — never silently
overwrite one candidate's output with another's.

---

## 2. Model Manifest

| # | Stage | Primary | Fallback / Bake-off | Notes |
|---|-------|---------|----------------------|-------|
| 1 | Extraction/mux | FFmpeg | — | Extract 48kHz stereo master + 16kHz mono copy for ASR |
| 2 | Source separation | UVR + BS-RoFormer | Demucs v4 `htdemucs_ft` | Run fallback automatically if primary output fails QC (see §4.2) |
| 3 | STT (Hindi) | IndicConformer Hindi (`ai4bharat/indic-conformer-600m-multilingual`) | IndicWhisper (AI4Bharat Vistaar fine-tune) | **Run both unconditionally for every clip in this experimentation phase** — do not treat IndicWhisper as "only on failure." See §6.1, this ordering is unresolved pending a real test. |
| 4 | Word timestamps | WhisperX | — | Forced alignment (wav2vec2-based) on top of whichever STT transcript wins §6.1 |
| 5 | Diarization | pyannote `community-1` | — | Verify exact HF checkpoint ID at implementation time (search `pyannote community-1` on Hugging Face) |
| 6 | Timeline merge | Custom Python/JSON layer | — | Merges STT text + word timestamps + speaker labels into canonical timeline; schema in §5 |
| 7 | Translation (Hindi→Bangla) | IndicTrans2 (`ai4bharat/indictrans2`) | — | Segment-level, one call per timeline segment |
| 8 | Translation cleanup | LLM + `glossary/mythology_glossary.json` | — | Enforce fixed mappings for deity names/epithets/Sanskrit terms; LLM call must be constrained to glossary terms, not free rewriting |
| 9 | TTS voice cloning (Bangla) | DhVaani 0.5 (`ARTPARK-IISc/DhVaani-0.5`) | Fish Speech S2 Pro, IndicF5 (`ai4bharat/IndicF5`), svara-tts-voiceclone-beta | **Run all four on the same validation segments before locking a default** — see §6.2. Do not silently pick DhVaani as final without running the emotional-delivery test. |
| 10 | Duration fit | TTS native duration control (DhVaani `speed` param) | FFmpeg `atempo`/time-stretch correction | Only invoke FFmpeg correction if native duration control misses target segment length by more than the tolerance in §4.3 |
| 11 | Audio remix | FFmpeg | — | Dubbed vocal (post duration-fit) + original BGM stem from stage 2 |
| 12 | Lip sync | LatentSync 1.6 | MuseTalk | Fallback triggers per §4.4 (compute/time budget or LatentSync failure on a given clip) |
| 13 | Final mux/export | FFmpeg | — | Sync corrected video + remixed audio → final deliverable |

For every model above whose exact checkpoint/repo ID is not 100% certain,
verify via web search or the model's own HF/GitHub page before first use —
do not guess an ID and silently fail. Record the verified ID in
`configs/pipeline.yaml` once confirmed.

---

## 3. Reference Materials Already Established

- **Known Sanskrit/tatsam ASR benchmark**: the Vedavani study (Sanskrit Vedic
  poetry ASR benchmark) found IndicWhisper leading on CER for Devanagari
  script, with plain Whisper large only marginally ahead on WER. No
  equivalent published benchmark exists yet for IndicConformer on this exact
  vocabulary — this is why §6.1 requires an explicit side-by-side test rather
  than trusting IndicConformer's "primary" label blindly.
- **DhVaani 0.5**: ZipVoice fine-tune, 123M params, flow-matching TTS, 27
  Indic languages, zero-shot voice cloning from a few seconds of reference
  audio + its transcript, 24kHz output. Does not have documented explicit
  disentangled emotion/timbre control the way some non-Indic models do —
  emotional fidelity depends entirely on how well it transfers prosody from
  the reference clip. This is unverified for this project's use case and
  must be tested per §6.2.
- **svara-tts-voiceclone-beta**: 19 Indic languages including Sanskrit,
  explicit emotion tags (`<laugh>`, `<yawn>`, `<angry>`, etc.), reference-swap
  fine-tuning for cloning stability, Orpheus-style discrete audio token
  architecture, GGUF/vLLM compatible. Candidate specifically for the
  emotional-delivery requirement if DhVaani underperforms on high-emotion
  lines.

---

## 4. Fallback Trigger Rules (must be encoded, not left to human judgment mid-run)

### 4.1 General principle
Every stage with a fallback must have an automatic QC check that decides
whether to fall back — the pipeline should not require a human to look at
output and decide. Where no numeric QC check is feasible yet (early
experimentation), flag the segment in `logs/needs_review.json` and continue
with the primary output rather than blocking the pipeline.

### 4.2 Source separation fallback
Trigger Demucs fallback if UVR+BS-RoFormer output vocal stem has:
- silence/near-silence (RMS below threshold) across >20% of segments where
  the diarization step (stage 5) expects speech, OR
- the process errors/crashes.
Define exact RMS threshold empirically on first 5 test clips; record the
chosen value in `configs/pipeline.yaml`.

### 4.3 Duration fit tolerance
If DhVaani's native duration control output is within ±150ms of the source
segment duration, accept as-is. Beyond that, apply FFmpeg `atempo` correction
(bounded to 0.85x–1.15x speed to avoid audible pitch/quality artifacts —
do not stretch further; if the required correction exceeds this bound,
flag the segment in `logs/needs_review.json` instead of forcing a bad stretch).

### 4.4 Lip sync fallback
Use MuseTalk instead of LatentSync 1.6 when:
- LatentSync errors/crashes on a clip, OR
- compute/time budget for the run is constrained (MuseTalk is the
  speed-oriented choice) — this is a runtime config flag, not a per-clip
  automatic decision; default to LatentSync 1.6 unless the flag is set.

---

## 5. Canonical Timeline JSON Schema (stage_05_timeline output)

```json
{
  "clip_id": "clip001",
  "source_language": "hi",
  "target_language": "bn",
  "segments": [
    {
      "segment_id": "clip001_seg001",
      "speaker_id": "SPEAKER_00",
      "start_ms": 1234,
      "end_ms": 3456,
      "source_text": "<Hindi transcript for this segment>",
      "source_text_engine": "indicconformer | indicwhisper",
      "words": [
        {"text": "...", "start_ms": 1234, "end_ms": 1400}
      ],
      "translated_text_raw": "<IndicTrans2 output>",
      "translated_text_final": "<post-glossary-cleanup output>",
      "tts_reference_audio": "data/stage_02_separated/clip001__vocal.wav#1234-3456",
      "tts_output_candidates": {
        "dhvaani": "data/stage_07_tts/clip001_seg001__tts__dhvaani.wav",
        "fishspeech": "data/stage_07_tts/clip001_seg001__tts__fishspeech.wav",
        "indicf5": "data/stage_07_tts/clip001_seg001__tts__indicf5.wav",
        "svara": "data/stage_07_tts/clip001_seg001__tts__svara.wav"
      },
      "duration_fit_ms": 2222,
      "flags": []
    }
  ]
}
```

`flags` holds strings like `"needs_review"`, `"duration_correction_exceeded_bound"`,
`"separation_fallback_used"` — anything that should surface in QC without
halting the run.

---

## 6. Validation Protocol (run before locking any "primary" choice as final)

### 6.1 STT bake-off (IndicConformer vs IndicWhisper)
- Select 5–10 dialogue segments with dense tatsam vocabulary (Sanskrit
  epithets, compound words, formal/shastric register lines — the kind of
  dialogue mythological serials are full of).
- Run both engines on each segment.
- Score by manual transcript comparison against ground truth (human-verified
  transcript of those segments) — character error rate is the more
  informative metric than word error rate for this vocabulary, per the
  Vedavani benchmark precedent.
- Whichever engine wins on this specific test set becomes primary for this
  project; update §2 manifest and `configs/pipeline.yaml` accordingly. Do
  not default to the order currently listed in §2 without running this.

### 6.2 TTS bake-off (tonality/emotional delivery)
- Select at least 2 reference segments per test: one neutral-register line,
  one high-emotion line (anger, grief, or theatrical declamation — common in
  mythological content).
- Run DhVaani, Fish Speech S2 Pro, IndicF5, and svara-tts-voiceclone-beta on
  each, cloning from the same reference audio.
- Evaluate: (a) speaker identity preservation, (b) whether emotional
  intensity of the reference carries into the Bangla output, (c) Bangla
  pronunciation naturalness, (d) duration control accuracy.
- The neutral-line result alone is not sufficient to pick a winner — a model
  that only performs well on flat delivery does not meet this project's
  stated tonality requirement.

### 6.3 Separation QC
- Spot-check UVR+BS-RoFormer output against Demucs on 3–5 clips with dense
  background score (common in mythological serials — these often have
  near-continuous orchestral/devotional background music under dialogue).

---

## 7. Definition of Done — Experimentation Phase

A clip is considered a complete successful pipeline run when:
1. All 13 stages have produced output with no unresolved `"needs_review"` flags, OR flags exist but were manually reviewed and accepted.
2. Final output audio has dubbed Bangla dialogue with background score intact and audible at approximately original levels.
3. Lip sync visually matches dubbed audio on manual review (no automated visual QC yet at this phase — human eyeball check is acceptable).
4. §6.1 and §6.2 bake-offs have been run at least once and their outcome recorded in `DECISIONS.md`, even if the outcome is "primary confirmed, no change."

Once 3–5 clips clear this bar consistently, the pipeline is ready to move
from single-clip experimentation to batch/episode-level processing — that
transition is out of scope for this document and should be a new task spec.

---

## 8. Explicit Non-Goals For This Phase

- No batch processing / queueing infrastructure yet.
- No language pairs other than Hindi→Bangla.
- No automated visual lip-sync QC scoring.
- No production deployment, hosting, or API wrapping.
- No fine-tuning of any model — this phase uses off-the-shelf checkpoints only.