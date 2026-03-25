"""Shared utilities for CAMB AI API integration."""

import os
import time

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CAMB_API_BASE = "https://client.camb.ai/apis"

# Language name -> CAMB integer ID mapping
CAMB_LANGUAGE_IDS = {
    "English": 1,
    "中文": 139,
    "简体中文": 139,
    "繁体中文": 146,
    "Japanese": 88,
    "Korean": 94,
    "French": 76,
    "Spanish": 54,
    "Polish": 109,
    "粤语": 145,
    "Cantonese": 145,
}


def get_api_key():
    key = os.getenv("CAMB_API_KEY", "")
    if not key:
        raise ValueError("CAMB_API_KEY environment variable is required for CambAI")
    return key


def poll_task(endpoint, task_id, interval=10, timeout=600):
    """Poll a CAMB AI async task until completion.

    Args:
        endpoint: The status endpoint path (e.g. '/dub', '/transcribe', '/translate')
        task_id: The task ID returned by the create call
        interval: Seconds between polls
        timeout: Max seconds to wait

    Returns:
        run_id on success

    Raises:
        RuntimeError on failure or timeout
    """
    api_key = get_api_key()
    url = f"{CAMB_API_BASE}/{endpoint.strip('/')}/{task_id}"
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise RuntimeError(f"CambAI task {task_id} timed out after {timeout}s")

        response = requests.get(url, headers={"x-api-key": api_key})
        response.raise_for_status()
        data = response.json()
        status = data.get("status", "")

        if status == "SUCCESS":
            run_id = data.get("run_id")
            if run_id is None:
                raise RuntimeError(f"CambAI task {task_id} succeeded but returned no run_id")
            logger.info(f"CambAI task {task_id} completed, run_id={run_id}")
            return run_id
        elif status == "PENDING":
            logger.debug(f"CambAI task {task_id} pending ({int(elapsed)}s elapsed)...")
            time.sleep(interval)
        else:
            raise RuntimeError(f"CambAI task {task_id} failed with status: {status}")


def get_camb_language_id(language_name):
    """Convert a language name to a CAMB AI integer language ID."""
    lang_id = CAMB_LANGUAGE_IDS.get(language_name)
    if lang_id is None:
        raise ValueError(
            f"Unsupported language for CambAI: {language_name}. "
            f"Supported: {list(CAMB_LANGUAGE_IDS.keys())}"
        )
    return lang_id
