import sys
import os

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal

app = modal.App("test-fishspeech-install")

fish_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1", "build-essential")
    .pip_install("torch>=2.4.0", "torchaudio>=2.4.0")
    .pip_install("git+https://github.com/fishaudio/fish-speech.git")
)

@app.function(image=fish_image, gpu="L4", timeout=600)
def check_fish_imports():
    import fish_speech
    print("fish_speech version / path:", fish_speech.__file__)
    import fish_speech.models.text2semantic.inference as t2s
    print("t2s inference imported successfully!")
    return True

def main():
    with app.run():
        res = check_fish_imports.remote()
    print("Fish speech install test:", res)

if __name__ == "__main__":
    main()
