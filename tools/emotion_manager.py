import os
import json
from pathlib import Path


def load_emotion_profile(folder):
    profile_path = Path(folder) / "emotion_profile.json"
    if not profile_path.exists():
        return {}
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_emotion_profile(folder, profile):
    profile_path = Path(folder) / "emotion_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def apply_emotion_to_tts(text, emotion):
    mapping = {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "surprised": "surprised",
    }
    return mapping.get(emotion, "neutral")
