#!/usr/bin/env python3
"""
Stage 07 — Translation: Hindi → Bangla (IndicTrans2)
======================================================
Reads:  data/stage_05_timeline/{clip_id}__timeline.json
Writes: data/stage_06_translated/{clip_id}__translated__raw.json
        (updates timeline, populating translated_text_raw per segment)

Model: IndicTrans2 (ai4bharat/indictrans2-indic-indic-1B, fallback to dist-320M)
       src: hin_Deva  →  tgt: ben_Beng

Translates segment-level source_text via Modal L4 GPU using IndicTransToolkit.
Glossary cleanup happens in Stage 08 — do NOT apply glossary here.

Usage:
  python scripts/07_translate.py --clip-id clip001
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows console
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

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("07_translate")
    logging.basicConfig(level=logging.INFO)

try:
    import yaml
except ImportError:
    yaml = None

# ── Paths & Config ────────────────────────────────────────────────────────────

CONFIG_PATH = Path("configs/pipeline.yaml")


def load_config() -> dict:
    if yaml is None:
        raise RuntimeError("yaml module is required on host to read config")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Modal App & Container Environment ─────────────────────────────────────────

app = modal.App("dubbing-stage-07-translate")

translate_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.2.0",
        # Pin transformers to 4.46.1: AI4Bharat's IndicTransTokenizer and
        # configuration_indictrans.py require older tokenizer internals
        # (_special_tokens_map init order) and transformers.onnx that were
        # removed/changed in transformers >= 4.47+.
        "transformers==4.46.1",
        "sentencepiece",
        "sacremoses",
        "nltk",
        "accelerate",
        "indic-nlp-library",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
    )
    .run_commands(
        "pip install git+https://github.com/VarunGumma/IndicTransToolkit.git",
        "python3 -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"",
    )
)


@app.function(
    gpu="L4",
    image=translate_image,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def translate_segments_modal(
    sentences: list[str],
    src_lang: str = "hin_Deva",
    tgt_lang: str = "ben_Beng",
    hf_repo: str = "ai4bharat/indictrans2-indic-indic-1B",
    batch_size: int = 8,
) -> list[str]:
    """Translates a list of sentences from src_lang to tgt_lang using IndicTrans2 on GPU."""
    import os
    import torch

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        from IndicTransToolkit.processor import IndicProcessor


    hf_token = os.environ.get("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading IndicTrans2 model ({hf_repo}) on {device} (auth token present: {bool(hf_token)})...")
    t0 = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True, token=hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_repo,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            token=hf_token,
        ).to(device)
    except Exception as exc:
        print(f"Failed to load primary model {hf_repo}: {exc}")
        if "indictrans2-indic-indic-1B" in hf_repo:
            fallback_repo = "ai4bharat/indictrans2-indic-indic-dist-320M"
            print(f"Switching to fallback model: {fallback_repo}...")
            tokenizer = AutoTokenizer.from_pretrained(fallback_repo, trust_remote_code=True, token=hf_token)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                fallback_repo,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                token=hf_token,
            ).to(device)
            hf_repo = fallback_repo
        else:
            raise

    model.eval()

    ip = IndicProcessor(inference=True)
    load_time = time.time() - t0
    print(f"Model and tokenizer ({hf_repo}) loaded successfully in {load_time:.2f}s.")

    translated_results: list[str] = []

    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        print(f"Translating batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size} ({len(chunk)} items)...")

        # Handle empty/whitespace strings gracefully
        valid_indices = [idx for idx, s in enumerate(chunk) if s and s.strip()]
        valid_sentences = [chunk[idx].strip() for idx in valid_indices]

        if not valid_sentences:
            translated_results.extend(["" for _ in chunk])
            continue

        # Preprocess batch
        preprocessed = ip.preprocess_batch(valid_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            preprocessed,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=256,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                use_cache=True,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
            )

        decoded = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        postprocessed = ip.postprocess_batch(decoded, lang=tgt_lang)

        # Map back to original chunk positions
        chunk_results = ["" for _ in chunk]
        for sub_idx, orig_idx in enumerate(valid_indices):
            chunk_results[orig_idx] = postprocessed[sub_idx]

        translated_results.extend(chunk_results)

    total_time = time.time() - t0
    print(f"Completed translation of {len(sentences)} sentences in {total_time:.2f}s ({load_time:.2f}s model load).")
    return translated_results


# ── Host Execution ────────────────────────────────────────────────────────────

def run_translation(clip_id: str, cfg: dict):
    stage_05_dir = Path(cfg["paths"]["stage_05"])
    stage_06_dir = Path(cfg["paths"]["stage_06"])
    stage_06_dir.mkdir(parents=True, exist_ok=True)

    input_timeline_file = stage_05_dir / f"{clip_id}__timeline.json"
    output_raw_file = stage_06_dir / f"{clip_id}__translated__raw.json"

    if not input_timeline_file.exists():
        raise FileNotFoundError(f"Input timeline JSON not found: {input_timeline_file}")

    logger.info(f"Loading timeline from {input_timeline_file}...")
    with open(input_timeline_file, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    segments = timeline_data.get("segments", [])
    if not segments:
        raise ValueError("No segments found in timeline JSON.")

    logger.info(f"Loaded {len(segments)} segments for translation.")

    trans_cfg = cfg.get("translation", {})
    hf_repo = trans_cfg.get("hf_repo", "ai4bharat/indictrans2-indic-indic-1B")
    src_lang = trans_cfg.get("src_lang", "hin_Deva")
    tgt_lang = trans_cfg.get("tgt_lang", "ben_Beng")
    batch_size = trans_cfg.get("batch_size", 8)

    source_texts = [seg.get("source_text", "") for seg in segments]

    logger.info(f"Dispatching {len(source_texts)} segments to Modal L4 GPU (model: {hf_repo}, {src_lang} -> {tgt_lang})...")
    start_time = time.time()

    with modal.enable_output():
        with app.run():
            translated_texts = translate_segments_modal.remote(
                sentences=source_texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                hf_repo=hf_repo,
                batch_size=batch_size,
            )

    elapsed = time.time() - start_time
    logger.info(f"Modal translation finished in {elapsed:.2f} seconds.")

    if len(translated_texts) != len(segments):
        raise RuntimeError(f"Expected {len(segments)} translations, got {len(translated_texts)}.")

    # Populate translated_text_raw in timeline
    for seg, trans in zip(segments, translated_texts):
        seg["translated_text_raw"] = trans

    # Save to stage_06_translated
    logger.info(f"Saving updated timeline to {output_raw_file}...")
    with open(output_raw_file, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False)

    # Print clean summary table
    print("\n" + "=" * 90)
    print(f"STAGE 07 TRANSLATION SUMMARY: {clip_id}")
    print(f"Model: {hf_repo} | {src_lang} -> {tgt_lang} | Segments: {len(segments)} | Time: {elapsed:.2f}s")
    print("=" * 90)
    print(f"{'Segment ID':<16} | {'Window (ms)':<14} | {'Hindi Source Text':<30} | {'Bangla Raw Translation'}")
    print("-" * 90)

    for seg in segments:
        seg_id = seg.get("segment_id", "")
        window = f"{seg.get('start_ms', 0)}-{seg.get('end_ms', 0)}"
        src = seg.get("source_text", "")
        raw = seg.get("translated_text_raw", "")
        
        # Truncate for pretty terminal table if long
        src_disp = (src[:27] + "...") if len(src) > 30 else src
        raw_disp = (raw[:35] + "...") if len(raw) > 38 else raw
        print(f"{seg_id:<16} | {window:<14} | {src_disp:<30} | {raw_disp}")

    print("=" * 90 + "\n")
    logger.info(f"[SUCCESS] Stage 07 raw translation written to {output_raw_file}")


def main():
    parser = argparse.ArgumentParser(description="Stage 07 — Translation (Hindi -> Bangla IndicTrans2)")
    parser.add_argument("--clip-id", default=None, help="Clip identifier (default: from pipeline.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    clip_id = args.clip_id or cfg.get("project", {}).get("clip_id_default", "clip001")

    run_translation(clip_id, cfg)


if __name__ == "__main__":
    main()
