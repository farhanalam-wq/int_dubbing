# Developer Setup

## Prerequisites

- **Python 3.14+** — already on PATH
- **ffmpeg** — install from https://ffmpeg.org/download.html and add to PATH
- **uv** — fast Python package manager (replaces pip/venv)
- **Modal account** — https://modal.com (GPU compute for all model stages)

---

## 1. Install uv (one-time)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Then add to PATH for current session:
$env:Path = "C:\Users\$env:USERNAME\.local\bin;$env:Path"
```

## 2. Set up local environment

```powershell
# Install local deps (audio I/O, ffmpeg bindings, loguru, yaml, etc.)
uv sync --system-certs

# Activate the venv (for running scripts directly)
.venv\Scripts\Activate.ps1
```

> Heavy ML deps (torch, transformers, whisperx, etc.) are NOT installed locally.
> They run inside Modal GPU containers. See pyproject.toml [gpu] optional group.

## 3. Set up Modal

```powershell
# Authenticate with your Modal account (one-time)
.venv\Scripts\modal.exe setup

# Verify login
.venv\Scripts\modal.exe profile current
```

Set secrets on Modal dashboard (https://modal.com/secrets) for:
- `HF_TOKEN` — HuggingFace token (required for pyannote diarization)
- `OPENAI_API_KEY` — for stage 8 translation cleanup LLM

## 4. Put test video in place

```powershell
copy assets\test_hindi_video.mp4 data\raw\clip001.mp4
```

---

## Running a Stage

### Stage 1 (local — FFmpeg only, no GPU needed)
```powershell
.venv\Scripts\Activate.ps1
python scripts/01_extract.py --clip-id clip001 --verbose
```

### Stages 2–13 (Modal GPU)
```powershell
# Deploy a stage to Modal and run it
modal run scripts/02_separate.py --clip-id clip001

# Or run locally with Modal's local execution mode (slower, for debugging)
modal run --detach scripts/02_separate.py --clip-id clip001
```

---

## Adding Dependencies

```powershell
# Add a local dep
uv add <package> --system-certs

# Add a GPU dep (goes in [gpu] optional group)
uv add --optional gpu <package> --system-certs
```

---

## uv Cheatsheet

| Command | What it does |
|---------|-------------|
| `uv sync --system-certs` | Install/update all deps from pyproject.toml |
| `uv add <pkg> --system-certs` | Add a new dependency |
| `uv remove <pkg>` | Remove a dependency |
| `uv run python scripts/01_extract.py` | Run script in the venv without activating |
| `uv lock --system-certs` | Regenerate uv.lock |
