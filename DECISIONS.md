# DECISIONS.md — Engineering Call Log

Every decision made without explicit user confirmation is recorded here.
Format: date · decision · rationale.

---

## 2026-09-08

### D-001 · Added `requirements.txt`
TASK.md does not mention a requirements file but one is necessary for
reproducibility. Created `requirements.txt` with all known Python dependencies
per the model manifest. Pinning is intentionally loose (major version only) at
this stage; tighten after first successful full-pipeline run.

### D-002 · Kept test video in `assets/`, not moved to `data/raw/`
`assets/test_hindi_video.mp4` was present at scaffold time. Per §1, raw inputs
belong in `data/raw/`. Did not move the file automatically to avoid silent
mutation of user data. User should copy/move it manually:
  `copy assets\test_hindi_video.mp4 data\raw\clip001.mp4`
The clip_id `clip001` is used as the default example throughout configs.

### D-003 · Script numbering follows model manifest (01–13), not data folder count
Data folders only go to stage_10; scripts go 01–13 because stages 4+5 both
write to `stage_04_aligned/` and stages 7+8 both write to `stage_06_translated/`.
Scripts are numbered to match §2 model manifest rows for unambiguous traceability.

### D-004 · `logs/needs_review.json` initialised as empty array `[]`
Schema is an append-only list of flag objects. Empty array is valid initial state.

### D-005 · Injected Windows system root certificates into `.venv` certifi bundle
On this machine's network environment, SSL interception/corporate root certificates
are required for outbound HTTPS connections (which caused uv `--system-certs` requirement).
Python's `certifi` package defaults to Mozilla CA roots and lacked the local Windows root
certificates, causing `modal token set` and API calls to fail with `CERTIFICATE_VERIFY_FAILED`.
Appended Windows Root Store certificates (`ssl.enum_certificates('ROOT')`) into
`.venv/Lib/site-packages/certifi/cacert.pem`. Modal authentication and API connectivity now succeed.

### D-006 · Stage 02 Modal L4 separation & Windows console encoding
Implemented Stage 02 source separation using Modal L4 GPU compute with `audio-separator`
(BS-RoFormer `model_bs_roformer_ep_317_sdr_12.9755.ckpt`) as primary and Demucs v4
(`htdemucs_ft`) as automatic QC fallback on >20% silence or crash. Added UTF-8 stdout/stderr
reconfiguration to bypass Windows `cp1252` encoding errors with Modal's Unicode checkmarks (`\u2713`).

### D-007 · Stage 03 Hindi STT Bake-off dual-engine configuration
Verified and configured both candidate Hindi ASR models per §6.1 bake-off requirements:
`ai4bharat/indic-conformer-600m-multilingual` (multilingual conformer via AutoModel) and
`openai/whisper-large-v3` (timestamped Hindi transcription). Both run concurrently on a Modal
L4 GPU container writing separate candidate outputs to `data/stage_03_transcript/`.
IndicConformer won decisively on shastric/tatsam Sanskrit vocabulary and zero hallucinations.

### D-008 · Stage 04 WhisperX word alignment on IndicConformer transcript
Employed WhisperX's wav2vec2 Hindi phoneme alignment model (`language_code="hi"`) on Modal L4
GPU to perform forced alignment on the winning IndicConformer transcript against the 16kHz mono
master audio. Achieved millisecond word timestamps across all 158 spoken words. Output saved to
`data/stage_04_aligned/clip001__aligned__words.json`.

### D-009 · Stage 05 Pyannote community-1 speaker diarization
Ran `pyannote/speaker-diarization-community-1` on Modal L4 GPU over the 48kHz master audio.
Successfully segmented speech into 27 discrete dialogue turns for single speaker `SPEAKER_00` in 4.69s.
Output saved to `data/stage_04_aligned/clip001__diarized__speakers.json`.

### D-010 · Stage 06 Canonical timeline schema merge
Merged IndicConformer text, WhisperX word timestamps, and Pyannote speaker turns into the
canonical 19-segment timeline conforming to §5 schema. Trimmed intro silence from the first
word using diarization speech onset (`10479ms`) and grouped words by 550ms acoustic pauses.
Output saved to `data/stage_05_timeline/clip001__timeline.json`.

### D-011 · Stage 07 IndicTrans2 1B execution & transformers pinning
Resolved IndicTrans2 environment compatibility by pinning `transformers==4.46.1` in the Modal container image. This cleanly satisfied `IndicTransTokenizer`'s internal attribute initialization order and legacy `transformers.onnx` imports without fragile monkey-patches. Ran the primary `ai4bharat/indictrans2-indic-indic-1B` model on Modal L4 GPU, translating 19 segments in 32.84s.
Output saved to `data/stage_06_translated/clip001__translated__raw.json`.

### D-012 · Stage 08 Translation cleanup & Shastric adaptation
Configured Gemini (`gemini-3.6-flash`) via direct REST API (zero extra dependencies) alongside `glossary/mythology_glossary.json` and local rule-based fallbacks.
The Shastric post-editing pass successfully:
1. Eliminated machine translation artifacts (e.g., IndicTrans2 transliteration *"ইন লাইফ"* replaced with proper Bangla *"জীবনে / জীবন তো"*).
2. Enforced Tatsama mythological register (*হৃদয়ে ধারণ*, *অনুভবই জীবন*, *সময়ের স্মরণ*, *সংঘর্ষ*).
3. Preserved canonical timing boundaries and dialogue length.
### D-013 · Stage 09 Multi-Engine TTS Bake-Off & DhVaani-0.5 resolution
Resolved DhVaani container environment on Modal L4:
1. Gated model authorized on Hugging Face using `HF_TOKEN` from `hf-secret`.
2. Added full upstream `k2-fsa/ZipVoice` dependencies (`vocos`, `pydub`, `lhotse`, `safetensors`, `cn2an`, `inflect`, `jieba`, `pypinyin`).
3. Resolved `torchaudio` v2.6+ backend incompatibility by redirecting `torchaudio.load` and `torchaudio.save` directly to `soundfile`.
4. Fixed ZipFormer relative positional encoding meta-tensor exception by passing `low_cpu_mem_usage=False` and sanitizing non-persistent buffers.
Executed bake-off on benchmark lines:
- `clip001_seg001` (neutral philosophical statement, target 1863ms): synthesized in 4.26s, output 1882ms (**delta: +19ms**, virtually perfect duration fit).
- `clip001_seg002` (theatrical declamation, target 4828ms): synthesized in 1.95s, output 4377ms (delta: -451ms).
Output audio saved to `data/stage_07_tts/clip001_seg001__tts__dhvaani.wav` and `clip001_seg002__tts__dhvaani.wav`.

### D-014 · Two-Stage Expressive Voice Conversion with OpenVoice v2 Tone Color Converter
To overcome the prosodic limitations of monolithic zero-shot TTS models on theatrical mythological lines:
1. Implemented a two-stage paradigm: Stage A (expressive, pause-annotated Bangla synthesis via Indic Parler-TTS) + Stage B (zero-shot tone color conversion via OpenVoice v2).
2. Resolved container dependencies on Modal L4: patched Cython build conflicts by pairing modern `faster-whisper` wheels with `--no-deps` OpenVoice installation, and called `ToneColorConverter.extract_se` directly to bypass external network VAD checks.
3. Extracted Krishna's reference speaker embedding from the isolated Hindi vocal track (`clip001_option_a_original_hindi.wav`) and converted the expressive Bangla speech in 0.20s (~60x real-time speedup).
4. Generated candidates across conversion strengths ($\tau=0.15, 0.30, 0.50$). Duration remained sample-locked to the 12.53s source timing without altering Shastric Bengali pronunciation.
Output saved to `data/stage_07_tts/clip001_option_a_cloned_voice.wav`.

### D-015 · Full-Video-Width Dubbed Audio Generation & Master Canvas Assembly
Scaled the verified Indic Parler-TTS + OpenVoice v2 tone color conversion pipeline across all 19 canonical dialogue segments:
1. Orchestrated unified container inference on Modal L4 GPU: synthesized all 19 segments with narrative-tailored style descriptions and applied zero-shot tone conversion conditioned on Krishna's vocal stem.
2. Pacing alignment: clamped duration scaling within `[0.85x, 1.20x]` via FFmpeg `atempo` so every segment comfortably matches its designated slot without bleeding into adjacent speech or music.
3. Master Canvas Assembly: built a 48kHz silent canvas matching the exact master clip duration (`93.6925s`, 4,497,241 frames). Composited all 19 dialogue segments at their canonical millisecond timestamps.
4. Verified silence: intro music (`0ms -> 10,479ms`) and mid-scene dramatic swells (`63,481ms -> 66,989ms`) contain zero digital audio (max amplitude = 0), ensuring the original BGM stem remains completely unobstructed.
Master track saved to `data/stage_07_tts/clip001__dubbed_vocal_full.wav`.

### D-016 · 6-Block Cohesive Dialogue Segmentation to Eliminate Voice Overlap & Erratic Pacing
Addressed auditory artifacts in the initial 19-segment full-width synthesis (accidental overlapping voices from segment spillage and jagged tempo changes from micro-fragmented inputs like 0.4s single-word turns):
1. Regrouped the 19 fragmented speech turns into 6 natural semantic thought-units bounded by actual pauses and music swells:
   - Block 1: Intro statement (10.48s - 12.34s)
   - Block 2: Theatrical monologue (Option A, 12.90s - 22.98s)
   - Block 3: Philosophical definition (23.54s - 31.08s)
   - Block 4: Aphorism (31.78s - 36.54s, merging 3 previous fragments)
   - Block 5: The human dilemma and time passage (37.52s - 63.48s, merging 6 previous fragments)
   - Block 6: Climactic resolution (66.99s - 93.72s, merging 4 previous fragments)
2. Enforced strict overlap prevention: every block's duration was paced so that `gap_to_next_s` is strictly positive (+0.08s to +11.59s), completely eliminating multi-speaker voice collisions.
3. Natural cadence: blocks 1, 3, and 4 play at 1.00x native speed; blocks 5 & 6 at 0.90x for solemn dramatic weight.
4. Preserved clean zero-amplitude silence during the 10.48s intro score and the 3.51s orchestral swell (63.48s - 66.99s).
Updated master vocal track saved to `data/stage_07_tts/clip001__dubbed_vocal_full.wav`.

### D-017 · Visual Breath-Phrase Anchoring & Pacing Synchronization
To resolve temporal drift where speech finished 1.5s to 4.0s before the actor closed his mouth on screen:
1. Segmented the monologue into 14 natural thought clauses anchored to WhisperX acoustic pause timestamps.
2. Preserved the actor's natural dramatic pauses (e.g. 1.7s contemplative pause after "অর্থাৎ," and 1.8s breath pause before "সত্য অনুধাবন করতে পারি না") as intentional silence on the vocal canvas.
3. Locked all dialogue synthesis to 1.00x un-stretched speed (zero `atempo` phase distortion) and leveled active speech RMS to ~0.078.
Updated master vocal track saved to `data/stage_07_tts/clip001__dubbed_vocal_full.wav`.

### D-018 · Stage 11 Dynamic BGM Remix with Sidechain Compression
Implemented Stage 11 in `scripts/11_remix.py`:
1. Highpass filtered vocal stem at 80Hz to eliminate proximity resonance.
2. Deployed FFmpeg `sidechaincompress` (threshold=0.035, ratio=2.2, attack=40ms, release=350ms) to duck the isolated BGM stem by ~3dB during speech delivery while releasing instantly during pauses and orchestral swells.
3. Summed inputs with unity gain (`normalize=0`) and brickwall peak limited to -1.0 dBFS (actual peak: -2.25 dBFS, 0.7720).
Output saved to `data/stage_09_remix/clip001__remix__final.wav`.

### D-019 · Stage 12 LatentSync 1.6 on Modal NVIDIA A100-80GB GPU
Configured ByteDance LatentSync 1.6 on Modal using `gpu="A100-80GB"` with `DeepCache` acceleration. Scaled compute from A10G to A100 80GB to eliminate timeout limits and accelerate 3D UNet latent diffusion across all 2,344 frames (completed in 11m 20s).

### D-020 · LatentSync Face-Fallback Patch for Non-Face Intro Scenes
Upstream ByteDance LatentSync crashed with `RuntimeError("Face not detected")` on scenery/title shots lacking a detected face (such as the 10.48s flute intro). Patched `ImageProcessor.affine_transform` and `LipsyncPipeline.restore_video` to gracefully pass through original video frames untouched whenever a face is absent, and interpolate momentary single-frame dropouts using previous facial landmarks.
Output saved to `data/stage_10_lipsync/clip001__lipsync__latentsync.mp4`.

### D-021 · Stage 13 Master Multiplexing & Quality Gate
Implemented Stage 13 in `scripts/13_mux.py` to replace the lip-synced video's temporary audio stream with the authoritative 48kHz stereo 320kbps remixed master WAV via stream-copy muxing (`-c:v copy -c:a aac -b:a 320k`). Output duration strictly verified at 93.76s matching source footage.
Final deliverable saved to `data/final/clip001__final.mp4`.

---

<!-- append new decisions below this line -->
