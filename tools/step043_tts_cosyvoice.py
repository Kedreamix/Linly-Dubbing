import os
import sys
import time
from loguru import logger
import torchaudio
import torch

from .utils import save_wav

sys.path.append("CosyVoice/third_party/Matcha-TTS")
sys.path.append("CosyVoice/")

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
except Exception as e:
    logger.warning(f"CosyVoice import failed: {e}")
    CosyVoice = None
    load_wav = None

model = None


def init_cosyvoice():
    load_model()


def download_cosyvoice():
    if not os.path.exists("models/TTS/CosyVoice-300M"):
        os.makedirs("models/TTS/CosyVoice-300M", exist_ok=True)
    logger.info("CosyVoice model directory ready.")


def load_model(model_path="models/TTS/CosyVoice-300M", device='auto'):
    global model
    if model is not None:
        return

    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Loading CosyVoice model from {model_path}")
    t_start = time.time()

    if not os.path.exists(model_path):
        download_cosyvoice()

    if CosyVoice is None:
        raise ImportError("CosyVoice package missing or incompatible.")

    model = CosyVoice(model_path)
    logger.info(f"CosyVoice loaded in {time.time() - t_start:.2f}s")


LANGUAGE_MAP = {
    "বাংলা": "bn",
    "English": "en",
    "中文": "zh",
    "हिन्दी": "hi",
    "日本語": "ja",
    "한국어": "ko",
    "Français": "fr",
    "Español": "es",
}


def tts(text, output_path, speaker_wav, model_name="models/TTS/CosyVoice-300M", device='auto', target_language='বাংলা'):
    global model

    if os.path.exists(output_path):
        logger.info(f"Output exists: {output_path}")
        return
    if model is None:
        load_model(model_name, device)

    if target_language not in LANGUAGE_MAP:
        raise ValueError(f"Unsupported language: {target_language}")

    for retry in range(3):
        try:
            prompt_speech_16k = load_wav(speaker_wav, 16000)
            lang_code = LANGUAGE_MAP[target_language]

            try:
                output = model.inference_cross_lingual(f"<|{lang_code}|>{text}", prompt_speech_16k)
            except Exception:
                output = model.inference(f"<|{lang_code}|>{text}", prompt_speech_16k)

            if isinstance(output, dict) and "tts_speech" in output:
                speech = output["tts_speech"]
            elif hasattr(output, "tts_speech"):
                speech = output.tts_speech
            else:
                raise ValueError("CosyVoice output format unsupported.")

            torchaudio.save(output_path, speech, 22050)
            logger.info(f"CosyVoice TTS done: {text}")
            return
        except Exception as e:
            logger.warning(f"CosyVoice failed: {e}")
            if retry == 2:
                raise
