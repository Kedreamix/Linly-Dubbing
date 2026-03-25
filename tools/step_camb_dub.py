"""End-to-end video dubbing via CAMB AI API.

Replaces the entire pipeline (audio separation, ASR, translation, TTS, synthesis)
with a single API call to CAMB AI's /dub endpoint.
"""

import os

import requests
from loguru import logger

from .camb_utils import CAMB_API_BASE, get_api_key, get_camb_language_id, poll_task


def dub_video(video_url, source_language, target_language, output_path=None):
    """Dub a video using CAMB AI's end-to-end dubbing API.

    Args:
        video_url: URL of the video (YouTube, Google Drive, or direct link)
        source_language: Source language name (e.g. 'English', '中文')
        target_language: Target language name (e.g. '中文', 'English')
        output_path: Where to save the dubbed video. If None, returns the URL.

    Returns:
        dict with 'video_url', 'audio_url', and 'transcript'
    """
    api_key = get_api_key()
    source_id = get_camb_language_id(source_language)
    target_id = get_camb_language_id(target_language)

    logger.info(f"CambAI Dub: {video_url} ({source_language} -> {target_language})")

    # Step 1: Submit dubbing task
    response = requests.post(
        f"{CAMB_API_BASE}/dub",
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "video_url": video_url,
            "source_language": source_id,
            "target_languages": [target_id],
        },
    )
    response.raise_for_status()
    task_id = response.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"CambAI Dub: API did not return a task_id: {response.json()}")
    logger.info(f"CambAI Dub: task_id={task_id}")

    # Step 2: Poll for completion
    run_id = poll_task("dub", task_id, interval=15, timeout=1200)

    # Step 3: Fetch result
    result_response = requests.get(
        f"{CAMB_API_BASE}/dub-result/{run_id}",
        headers={"x-api-key": api_key},
    )
    result_response.raise_for_status()
    result = result_response.json()

    dubbed_video_url = result.get("video_url")
    logger.info(f"CambAI Dub complete: {dubbed_video_url}")

    # Step 4: Download if output_path specified
    if output_path and dubbed_video_url:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        video_data = requests.get(dubbed_video_url)
        video_data.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(video_data.content)
        logger.info(f"CambAI Dub: saved to {output_path}")

    return result


if __name__ == "__main__":
    result = dub_video(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        source_language="English",
        target_language="中文",
        output_path="playground/test_camb_dub.mp4",
    )
    print(result)
