"""Speech-to-text transcription via CAMB AI API."""

import os

import requests
from loguru import logger

from .camb_utils import CAMB_API_BASE, get_api_key, get_camb_language_id, poll_task

# Default source language for transcription
DEFAULT_LANGUAGE = "English"


def camb_transcribe_audio(wav_path, language=DEFAULT_LANGUAGE):
    """Transcribe an audio file using CAMB AI's transcription API.

    Args:
        wav_path: Path to the audio WAV file
        language: Language name (e.g. 'English', '中文')

    Returns:
        list of dicts with 'start', 'end', 'text', 'speaker' keys
        (same format as WhisperX/FunASR output)
    """
    api_key = get_api_key()
    lang_id = get_camb_language_id(language)

    logger.info(f"CambAI Transcribe: {wav_path} (language={language})")

    # Step 1: Submit transcription task
    with open(wav_path, "rb") as f:
        response = requests.post(
            f"{CAMB_API_BASE}/transcribe",
            headers={"x-api-key": api_key},
            files={"media_file": (os.path.basename(wav_path), f)},
            data={"language": lang_id},
        )
    response.raise_for_status()
    task_id = response.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"CambAI Transcribe: API did not return a task_id: {response.json()}")
    logger.info(f"CambAI Transcribe: task_id={task_id}")

    # Step 2: Poll for completion
    run_id = poll_task("transcribe", task_id, interval=10, timeout=600)

    # Step 3: Fetch result
    result_response = requests.get(
        f"{CAMB_API_BASE}/transcription-result/{run_id}",
        headers={"x-api-key": api_key},
    )
    result_response.raise_for_status()
    transcript = result_response.json()

    # Normalize to expected format: list of {start, end, text, speaker}
    normalized = []
    for segment in transcript:
        normalized.append({
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "text": segment.get("text", "").strip(),
            "speaker": segment.get("speaker", "SPEAKER_00"),
        })

    logger.info(f"CambAI Transcribe: {len(normalized)} segments")
    return normalized


if __name__ == "__main__":
    result = camb_transcribe_audio("playground/test_audio.wav", language="English")
    for seg in result[:5]:
        print(seg)
