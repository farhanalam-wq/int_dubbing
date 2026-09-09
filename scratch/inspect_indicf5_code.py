import os
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal

app = modal.App("inspect-indicf5-code-part2")
image = modal.Image.debian_slim(python_version="3.10").pip_install("huggingface_hub")

@app.function(image=image, secrets=[modal.Secret.from_name("hf-secret")])
def get_code():
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    p = hf_hub_download("ai4bharat/IndicF5", "model.py", token=token)
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def main():
    with app.run():
        code = get_code.remote()
    print("=== IndicF5 model.py lines 40+ ===")
    lines = code.split("\n")
    print("\n".join(lines[35:100]))

if __name__ == "__main__":
    main()
