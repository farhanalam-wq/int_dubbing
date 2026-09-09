#!/usr/bin/env python3
"""
Stage 08 — Translation Cleanup & Shastric Adaptation (LLM + Mythology Glossary)
================================================================================
Reads:  data/stage_06_translated/{clip_id}__translated__raw.json
        glossary/mythology_glossary.json
Writes: data/stage_06_translated/{clip_id}__translated__final.json
        (updates timeline, filling translated_text_final per segment)

Refines raw machine translation (IndicTrans2) by:
1. Enforcing Shastric/Tatsama vocabulary from glossary/mythology_glossary.json.
2. Cleaning translation artifacts (e.g. English transliterations like "ইন লাইফ").
3. Elevating tone to Puranic/Mahabharat register without expanding syllable count.

Usage:
  python scripts/08_cleanup.py --clip-id clip001
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
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

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("08_cleanup")
    logging.basicConfig(level=logging.INFO)

try:
    import yaml
except ImportError:
    yaml = None


def load_env(env_path: Path = Path(".env")):
    """Load key-value pairs from .env if present into os.environ without requiring external packages."""
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")


def load_config(config_path: Path = Path("configs/pipeline.yaml")) -> dict:
    if yaml is None:
        raise RuntimeError("yaml module is required to read config")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_glossary(glossary_path: Path = Path("glossary/mythology_glossary.json")) -> dict:
    if not glossary_path.exists():
        logger.warning(f"Glossary not found at {glossary_path}")
        return {}
    with open(glossary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("terms", {})


def rule_based_cleanup(raw_text: str, glossary: dict) -> str:
    """Fallback rule-based cleanup in case LLM is unavailable."""
    text = raw_text

    # 1. Clean known IndicTrans2 transliteration artifacts
    artifact_replacements = [
        (r"\bইন\s+লাইফ\b", "জীবনে"),
        (r"\bইন\s+লাইফ\s+তাই\b", "জীবন তো"),
    ]
    for pattern, repl in artifact_replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 2. Check for Hindi glossary terms and substitute
    for hi_term, info in glossary.items():
        bn_term = info.get("bangla", "")
        if not bn_term:
            continue
        if hi_term in text:
            text = text.replace(hi_term, bn_term)

    return text.strip()


def call_gemini(
    segments_payload: list[dict],
    glossary: dict,
    api_key: str,
    model_name: str = "gemini-3.6-flash",
    system_prompt: str = "",
) -> dict[str, str]:
    """Calls Google Gemini API using urllib (zero extra dependencies)."""
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # Format glossary terms for context
    glossary_context = "\n".join(
        [f"- {hi} -> {meta.get('bangla', '')} ({meta.get('notes', '')})" for hi, meta in glossary.items()]
    )

    prompt = f"""{system_prompt}

### Glossary Constraints (STRICTLY ENFORCE these Bangla equivalents for deity names and Sanskrit/Tatsama terms):
{glossary_context}

### Additional Specific Rule:
- Watch out for English phonetic hallucinations from machine translation (e.g. "ইন লাইফ" MUST be translated properly to "জীবনে" or "জীবন তো").

### Input Segments:
{json.dumps(segments_payload, ensure_ascii=False, indent=2)}

### Output Format:
Return a valid JSON array of objects with keys "segment_id" and "translated_text_final".
Example:
[
  {{"segment_id": "clip001_seg001", "translated_text_final": "ভবিষ্যতের আরেকটি নাম"}},
  ...
]
Return ONLY the raw JSON array, without markdown code fences or conversational filler.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        }
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            resp_body = resp.read().decode("utf-8")
            resp_json = json.loads(resp_body)
            raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            
            # Parse JSON from model output
            clean_str = raw_text.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()

            parsed = json.loads(clean_str)
            if isinstance(parsed, list):
                return {item["segment_id"]: item["translated_text_final"].strip() for item in parsed if "segment_id" in item and "translated_text_final" in item}
            elif isinstance(parsed, dict) and "segments" in parsed:
                return {item["segment_id"]: item["translated_text_final"].strip() for item in parsed["segments"]}
            return {}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        logger.error(f"Gemini API HTTP {e.code} Error: {err_msg}")
        raise
    except Exception as e:
        logger.error(f"Gemini API request failed: {e}")
        raise


def run_cleanup(clip_id: str, cfg: dict):
    input_file = Path(f"data/stage_06_translated/{clip_id}__translated__raw.json")
    output_file = Path(f"data/stage_06_translated/{clip_id}__translated__final.json")
    glossary_path = Path("glossary/mythology_glossary.json")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}. Please run Stage 07 first.")

    logger.info(f"Loading raw translations from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    segments = timeline_data.get("segments", [])
    logger.info(f"Loaded {len(segments)} segments for Shastric cleanup.")

    glossary = load_glossary(glossary_path)
    logger.info(f"Loaded {len(glossary)} glossary terms from {glossary_path}.")

    cleanup_cfg = cfg.get("cleanup", {})
    provider = cleanup_cfg.get("llm_provider", "gemini").lower()
    model_name = cleanup_cfg.get("llm_model", "gemini-2.0-flash")
    api_key_env = cleanup_cfg.get("api_key_env", "GEMINI_API_KEY")
    system_prompt = cleanup_cfg.get("system_prompt", "")

    api_key = os.environ.get(api_key_env, "").strip()
    logger.info(f"Cleanup provider: {provider} | Model: {model_name} | Key env: {api_key_env} (set: {bool(api_key)})")

    # Prepare payload for LLM
    segments_payload = [
        {
            "segment_id": seg["segment_id"],
            "source_text_hi": seg.get("source_text", ""),
            "raw_translation_bn": seg.get("translated_text_raw", ""),
        }
        for seg in segments
    ]

    cleaned_map: dict[str, str] = {}

    if api_key and provider == "gemini":
        try:
            logger.info(f"Sending {len(segments)} segments to Gemini ({model_name}) for Shastric refinement...")
            t0 = time.time()
            cleaned_map = call_gemini(
                segments_payload=segments_payload,
                glossary=glossary,
                api_key=api_key,
                model_name=model_name,
                system_prompt=system_prompt,
            )
            logger.info(f"Gemini returned {len(cleaned_map)} refined segments in {time.time()-t0:.2f}s.")
        except Exception as e:
            logger.warning(f"Gemini call failed ({e}). Falling back to rule-based cleanup...")
    else:
        if not api_key:
            logger.warning(f"No API key found in ${api_key_env}. Using rule-based Shastric cleanup.")
        else:
            logger.warning(f"Provider '{provider}' not configured for direct REST. Using rule-based cleanup.")

    # Populate translated_text_final
    for seg in segments:
        seg_id = seg["segment_id"]
        raw_text = seg.get("translated_text_raw", "")
        if seg_id in cleaned_map and cleaned_map[seg_id]:
            seg["translated_text_final"] = cleaned_map[seg_id]
        else:
            seg["translated_text_final"] = rule_based_cleanup(raw_text, glossary)

    # Save to data/stage_06_translated/{clip_id}__translated__final.json
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved final Shastric translations to {output_file}")

    # Display comparison table
    print("\n" + "=" * 110)
    print(f"STAGE 08 TRANSLATION CLEANUP & SHASTRIC ADAPTATION: {clip_id}")
    print("=" * 110)
    print(f"{'Seg ID':<15} | {'Hindi Source Text':<28} | {'Raw Translation':<28} | {'Final Shastric Bangla'}")
    print("-" * 110)
    for seg in segments:
        seg_id = seg["segment_id"]
        hi = seg.get("source_text", "")
        raw = seg.get("translated_text_raw", "")
        final = seg.get("translated_text_final", "")
        hi_d = (hi[:25] + "...") if len(hi) > 28 else hi
        raw_d = (raw[:25] + "...") if len(raw) > 28 else raw
        print(f"{seg_id:<15} | {hi_d:<28} | {raw_d:<28} | {final}")
    print("=" * 110 + "\n")
    logger.info(f"[SUCCESS] Stage 08 complete: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Stage 08 — Translation Cleanup & Shastric Adaptation")
    parser.add_argument("--clip-id", default=None, help="Clip ID (default: from pipeline.yaml)")
    args = parser.parse_args()

    load_env()
    cfg = load_config()
    clip_id = args.clip_id or cfg.get("project", {}).get("clip_id_default", "clip001")

    run_cleanup(clip_id, cfg)


if __name__ == "__main__":
    main()
