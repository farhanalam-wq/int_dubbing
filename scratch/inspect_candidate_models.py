import sys
import os

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"

import os
import json
from pathlib import Path
import modal

app = modal.App("inspect-candidate-models")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "huggingface_hub",
        "transformers",
        "torch",
        "requests",
    )
)

@app.function(
    image=image,
    timeout=180,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def inspect_candidates():
    from huggingface_hub import HfApi, hf_hub_download
    import os

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    results = {}
    repos = [
        "ai4bharat/IndicF5",
        "fishaudio/s2-pro",
        "kenpath/svara-tts-voiceclone-beta",
    ]

    for r in repos:
        repo_info = {"repo": r}
        try:
            m_info = api.model_info(r)
            repo_info["pipeline_tag"] = m_info.pipeline_tag
            repo_info["tags"] = m_info.tags
            repo_info["files"] = [f.rfilename for f in m_info.siblings]
            
            # Download README
            if "README.md" in repo_info["files"]:
                try:
                    p = hf_hub_download(r, "README.md", token=token)
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        repo_info["readme"] = f.read()
                except Exception as e:
                    repo_info["readme_err"] = str(e)

            # Download config.json
            if "config.json" in repo_info["files"]:
                try:
                    p = hf_hub_download(r, "config.json", token=token)
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        repo_info["config"] = json.load(f)
                except Exception as e:
                    repo_info["config_err"] = str(e)

        except Exception as e:
            repo_info["error"] = str(e)

        results[r] = repo_info

    return results

def main():
    with modal.enable_output():
        with app.run():
            res = inspect_candidates.remote()

    out_p = Path("scratch/candidate_models_inspection.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("Saved candidate inspection to:", out_p)

if __name__ == "__main__":
    main()
