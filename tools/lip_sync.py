import os
import shutil
import subprocess
from pathlib import Path
from loguru import logger


def _copy_fallback(video_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(video_path, output_path)
    return output_path


def apply_lipsync_if_available(video_path, audio_path, output_path):
    """Best-effort lip-sync integration. If no dedicated model is installed, fallback to original video."""
    if not os.path.exists(video_path):
        return None
    if not os.path.exists(audio_path):
        return None

    candidates = [
        ("wav2lip_inference", ["python", "-m", "wav2lip_inference", "--face", video_path, "--audio", audio_path, "--out", output_path]),
        ("liveportrait_inference", ["python", "-m", "liveportrait_inference", "--video", video_path, "--audio", audio_path, "--out", output_path]),
    ]

    for name, cmd in candidates:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(output_path):
                logger.info(f"Lip-sync completed via {name}")
                return output_path
        except Exception:
            continue

    logger.info("No lip-sync engine available; preserving original video timing and audio output.")
    return _copy_fallback(video_path, output_path)
