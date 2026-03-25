"""Translation via CAMB AI API."""

import requests
from loguru import logger

from .camb_utils import CAMB_API_BASE, get_api_key, get_camb_language_id, poll_task


def camb_translate(texts, source_language, target_language):
    """Translate a list of texts using CAMB AI's translation API.

    Args:
        texts: List of strings to translate
        source_language: Source language name (e.g. 'English')
        target_language: Target language name (e.g. '简体中文')

    Returns:
        List of translated strings (same order as input)
    """
    api_key = get_api_key()
    source_id = get_camb_language_id(source_language)
    target_id = get_camb_language_id(target_language)

    logger.info(f"CambAI Translate: {len(texts)} texts ({source_language} -> {target_language})")

    # Step 1: Submit translation task
    response = requests.post(
        f"{CAMB_API_BASE}/translate",
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "source_language": source_id,
            "target_language": target_id,
            "texts": texts,
        },
    )
    response.raise_for_status()
    task_id = response.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"CambAI Translate: API did not return a task_id: {response.json()}")
    logger.info(f"CambAI Translate: task_id={task_id}")

    # Step 2: Poll for completion
    run_id = poll_task("translate", task_id, interval=5, timeout=300)

    # Step 3: Fetch result
    result_response = requests.get(
        f"{CAMB_API_BASE}/translation-result/{run_id}",
        headers={"x-api-key": api_key},
    )
    result_response.raise_for_status()
    result = result_response.json()

    translated_texts = result.get("texts", [])
    # Ensure output length matches input to prevent IndexError downstream
    if len(translated_texts) < len(texts):
        logger.warning(f"CambAI Translate: got {len(translated_texts)} translations for {len(texts)} inputs, padding")
        translated_texts.extend([""] * (len(texts) - len(translated_texts)))
    logger.info(f"CambAI Translate: got {len(translated_texts)} translations")
    return translated_texts


def camb_response(texts, source_language="English", target_language="简体中文"):
    """Wrapper matching the interface used by step030_translation.py.

    For CambAI, we batch-translate all texts at once (more efficient than one-by-one).
    """
    return camb_translate(texts, source_language, target_language)


if __name__ == "__main__":
    results = camb_translate(
        ["Hello, how are you?", "The weather is nice today."],
        source_language="English",
        target_language="简体中文",
    )
    for r in results:
        print(r)
