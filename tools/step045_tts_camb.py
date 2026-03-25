import os
import time

import requests
from loguru import logger

from .camb_utils import CAMB_API_BASE, get_api_key

# Mapping from Chinese language names to CAMB AI BCP-47 codes
LANGUAGE_MAP = {
    "中文": "zh-cn",
    "English": "en-us",
    "Japanese": "ja-jp",
    "Korean": "ko-kr",
    "French": "fr-fr",
    "Spanish": "es-es",
    "粤语": "zh-cn",  # Cantonese fallback to Mandarin
    "Polish": "pl-pl",
}

# Cache for cloned voice IDs per speaker
_voice_cache = {}


def _clone_voice(speaker_wav, api_key):
    """Clone a voice from reference audio and return the voice_id."""
    if speaker_wav in _voice_cache:
        return _voice_cache[speaker_wav]

    with open(speaker_wav, "rb") as f:
        response = requests.post(
            f"{CAMB_API_BASE}/create-custom-voice",
            headers={"x-api-key": api_key},
            files={"file": (os.path.basename(speaker_wav), f)},
            data={
                "voice_name": f"cloned-{os.path.basename(speaker_wav)}",
                "gender": 1,
                "language": 1,  # English as default
                "enhance_audio": "true",
            },
        )

    response.raise_for_status()
    voice_id = response.json().get("voice_id")
    if voice_id:
        _voice_cache[speaker_wav] = voice_id
        logger.info(f"CambAI: Cloned voice {voice_id} from {speaker_wav}")
    return voice_id


def tts(text, output_path, speaker_wav=None, target_language="中文"):
    if os.path.exists(output_path):
        logger.info(f"TTS {text} already exists")
        return

    api_key = get_api_key()
    camb_lang = LANGUAGE_MAP.get(target_language, "en-us")

    # Try to clone voice from speaker reference audio
    voice_id = 147320  # default voice
    if speaker_wav and os.path.exists(speaker_wav):
        try:
            cloned_id = _clone_voice(speaker_wav, api_key)
            if cloned_id:
                voice_id = cloned_id
        except Exception as e:
            logger.warning(f"CambAI: Voice cloning failed, using default voice: {e}")

    payload = {
        "text": text,
        "voice_id": voice_id,
        "language": camb_lang,
        "speech_model": "mars-flash",
        "output_configuration": {"format": "wav"},
    }

    for retry in range(3):
        try:
            response = requests.post(
                f"{CAMB_API_BASE}/tts-stream",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

            if not response.content or len(response.content) < 100:
                raise RuntimeError(f"CambAI TTS returned empty or invalid audio ({len(response.content)} bytes)")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"CambAI TTS: {text}")
            return
        except Exception as e:
            logger.warning(f"CambAI TTS failed (attempt {retry + 1}): {e}")
            time.sleep(1)


if __name__ == "__main__":
    tts("Hello, this is a test.", "playground/test_camb.wav", target_language="English")
