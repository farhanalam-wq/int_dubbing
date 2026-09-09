import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal

app = modal.App("test-bakeoff-models")

# Image with all necessary TTS & audio packages
bakeoff_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "transformers>=4.40.0",
        "soundfile",
        "scipy",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "safetensors",
        "loguru",
        "f5-tts",
        "snac",
        "vocos",
        "encodec",
    )
)

@app.function(
    gpu="L4",
    image=bakeoff_image,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def test_load_models():
    import os
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    status = {}

    # 1. Test IndicF5
    try:
        print("Testing IndicF5 load...")
        import contextlib
        try:
            import accelerate.big_modeling
            accelerate.big_modeling.init_empty_weights = contextlib.nullcontext
        except Exception as e:
            print("Accelerate patch note:", e)

        m_indic = AutoModel.from_pretrained(
            "ai4bharat/IndicF5",
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            token=hf_token
        )
        status["indicf5"] = {
            "loaded": True,
            "class": str(type(m_indic)),
            "dir_methods": [m for m in dir(m_indic) if not m.startswith("_")][:15],
        }
    except Exception as e:
        status["indicf5"] = {"loaded": False, "error": str(e)}

    # 2. Test Svara TTS
    try:
        print("Testing Svara TTS load...")
        cfg = AutoConfig.from_pretrained("kenpath/svara-tts-voiceclone-beta", token=hf_token)
        tok = AutoTokenizer.from_pretrained("kenpath/svara-tts-voiceclone-beta", token=hf_token)
        status["svara"] = {
            "loaded": True,
            "model_type": cfg.model_type,
            "architectures": cfg.architectures,
            "vocab_size": len(tok),
        }
    except Exception as e:
        status["svara"] = {"loaded": False, "error": str(e)}

    # 3. Test Fish Speech
    try:
        print("Testing Fish Speech S2 Pro config & tokenizer...")
        cfg = AutoConfig.from_pretrained("fishaudio/s2-pro", token=hf_token, trust_remote_code=True)
        status["fishspeech"] = {
            "loaded": True,
            "model_type": getattr(cfg, "model_type", None),
            "architectures": getattr(cfg, "architectures", None),
        }
    except Exception as e:
        status["fishspeech"] = {"loaded": False, "error": str(e)}

    return status

def main():
    with modal.enable_output():
        with app.run():
            res = test_load_models.remote()
    print("Model load status:")
    import json
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
