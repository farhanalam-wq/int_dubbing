import io
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"

import modal

app = modal.App("openvoice-tone-conversion")

# Modal Container Image with OpenVoice v2 dependencies
openvoice_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "soundfile",
        "scipy",
        "librosa",
        "silero-vad",
        "wavmark",
        "huggingface_hub",
        "pydub",
        "faster-whisper>=1.0.0",
        "whisper_timestamped",
        "inflect",
        "eng_to_ipa",
        "jieba",
        "cn2an",
        "pypinyin",
        "unidic_lite",
        "unidecode",
        "jamo",
    )
    .run_commands("pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git")
)


@app.function(
    gpu="L4",
    image=openvoice_image,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def convert_tone_modal(
    source_wav_bytes: bytes,
    target_ref_wav_bytes: bytes,
    tau_values: list[float] = [0.3],
) -> dict[str, bytes]:
    """
    Zero-shot tone color conversion using OpenVoice v2.
    - target_ref_wav: Original actor's vocal performance (Krishna Hindi monologue).
    - source_wav: Expressive Bangla synthesized speech (Indic Parler with prosodic pauses).
    - tau_values: Conversion strength controls (0.15 = subtle, 0.3 = standard, 0.5 = strong timbre match).
    """
    import os
    import tempfile
    import torch
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[OpenVoice] Initializing ToneColorConverter on {device}...")

    # 1. Download official OpenVoiceV2 converter weights from Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    config_path = hf_hub_download(
        repo_id="myshell-ai/OpenVoiceV2",
        filename="config.json",
        subfolder="converter",
        token=hf_token,
    )
    checkpoint_path = hf_hub_download(
        repo_id="myshell-ai/OpenVoiceV2",
        filename="checkpoint.pth",
        subfolder="converter",
        token=hf_token,
    )

    print(f"[OpenVoice] Loaded converter config: {config_path}")
    print(f"[OpenVoice] Loaded converter checkpoint: {checkpoint_path}")

    converter = ToneColorConverter(config_path, device=device)
    converter.load_ckpt(checkpoint_path)

    # Patch torch.hub to trust repos non-interactively
    try:
        import torch.hub
        torch.hub._check_repo_is_trusted = lambda *args, **kwargs: True
    except Exception:
        pass

    # 2. Write input audio to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "source_bangla.wav")
        ref_path = os.path.join(tmpdir, "target_krishna.wav")

        with open(src_path, "wb") as f:
            f.write(source_wav_bytes)
        with open(ref_path, "wb") as f:
            f.write(target_ref_wav_bytes)

        # 3. Directly extract speaker embeddings using the neural converter's reference encoder
        print("[OpenVoice] Extracting target speaker embedding (Krishna Hindi vocal)...")
        target_se = converter.extract_se([ref_path])

        print("[OpenVoice] Extracting source speaker embedding (Expressive Bangla audio)...")
        source_se = converter.extract_se([src_path])

        # 4. Perform Tone Color Conversion for each tau value
        results = {}
        for tau in tau_values:
            t0 = time.time()
            out_filename = f"converted_tau_{tau:.2f}.wav"
            out_path = os.path.join(tmpdir, out_filename)

            print(f"[OpenVoice] Running ToneColorConverter (tau={tau:.2f})...")
            converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=out_path,
                tau=tau,
            )
            elapsed = time.time() - t0
            print(f"[OpenVoice] Conversion completed in {elapsed:.2f}s (tau={tau:.2f})")

            with open(out_path, "rb") as f:
                results[f"tau_{tau:.2f}"] = f.read()

        return results


def main():
    src_audio_path = Path("data/stage_07_tts/clip001_option_a_annotated.wav")
    ref_audio_path = Path("data/stage_07_tts/clip001_option_a_original_hindi.wav")
    output_dir = Path("data/stage_07_tts")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not src_audio_path.exists():
        raise FileNotFoundError(f"Source expressive audio not found: {src_audio_path}")
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference actor audio not found: {ref_audio_path}")

    print("=" * 80)
    print("RUNNING ZERO-SHOT TONE COLOR CONVERSION (TWO-STAGE DUBBING PROTOTYPE)")
    print(f"Source (Expressive Bangla): {src_audio_path}")
    print(f"Reference (Krishna Hindi): {ref_audio_path}")
    print("=" * 80)

    with open(src_audio_path, "rb") as f:
        src_bytes = f.read()
    with open(ref_audio_path, "rb") as f:
        ref_bytes = f.read()

    # Test conversion with tau=0.15, 0.30, 0.50
    tau_test = [0.15, 0.30, 0.50]

    with modal.enable_output():
        with app.run():
            converted_outputs = convert_tone_modal.remote(
                source_wav_bytes=src_bytes,
                target_ref_wav_bytes=ref_bytes,
                tau_values=tau_test,
            )

    print("\nSaving converted outputs to data/stage_07_tts/...")
    saved_files = []
    for tag, wav_data in converted_outputs.items():
        out_file = output_dir / f"clip001_option_a_cloned_{tag}.wav"
        with open(out_file, "wb") as f:
            f.write(wav_data)
        saved_files.append(out_file)
        print(f"[SAVED] {out_file} ({len(wav_data)} bytes)")

    # Also make a convenient primary link
    primary_file = output_dir / "clip001_option_a_cloned_voice.wav"
    if "tau_0.30" in converted_outputs:
        with open(primary_file, "wb") as f:
            f.write(converted_outputs["tau_0.30"])
        print(f"[SAVED] Primary candidate: {primary_file}")

    print("\n[SUCCESS] Voice conversion completed successfully!")


if __name__ == "__main__":
    main()
