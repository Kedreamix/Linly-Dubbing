import os
import subprocess
import sys


def install_compatible_packages():
    print("Installing compatible packages for Python 3.13 + Colab...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "numpy==2.1.3 scipy==1.14.1 librosa==0.10.2 soundfile==0.12.0",
    ])
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
    ])
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "gradio==4.44.0 transformers==4.48.0 accelerate==1.1.0 diffusers==0.31.0 modelscope==1.9.0 funasr==1.0.0 edge-tts==7.0.0 yt-dlp==2025.1.0",
    ])
    print("Done.")


if __name__ == "__main__":
    install_compatible_packages()
