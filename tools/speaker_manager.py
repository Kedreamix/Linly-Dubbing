import os
import json
import shutil
from pathlib import Path
from loguru import logger


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_speaker_profiles(folder):
    """Create a lightweight speaker profile map from diarized ASR output."""
    speaker_dir = Path(folder) / "SPEAKER"
    ensure_dir(speaker_dir)

    profile_path = Path(folder) / "speaker_profiles.json"
    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    profiles = {}
    transcript_path = Path(folder) / "translation.json"
    if transcript_path.exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        for line in transcript:
            speaker = str(line.get("speaker", "SPEAKER_00"))
            if speaker not in profiles:
                profiles[speaker] = {
                    "speaker_id": speaker,
                    "voice_prompt": str(Path(folder) / "SPEAKER" / f"{speaker}.wav"),
                    "gender": "unknown",
                    "emotion": "neutral",
                    "style": "default",
                }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    return profiles


def assign_speaker_voice(folder, speaker_id, voice_path=None):
    """Persistent speaker-to-voice mapping."""
    profile_file = Path(folder) / "speaker_profiles.json"
    if not profile_file.exists():
        build_speaker_profiles(folder)

    with open(profile_file, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    speaker = str(speaker_id)
    if speaker not in profiles:
        profiles[speaker] = {
            "speaker_id": speaker,
            "voice_prompt": voice_path or str(Path(folder) / "SPEAKER" / f"{speaker}.wav"),
            "gender": "unknown",
            "emotion": "neutral",
            "style": "default",
        }
    else:
        if voice_path:
            profiles[speaker]["voice_prompt"] = voice_path

    with open(profile_file, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    return profiles[speaker]


def detect_emotion_from_text(text):
    """Simple lightweight emotion heuristic for dubbing."""
    lower = text.lower()
    if any(k in lower for k in ["angry", "furious", "hate", "no way", "damn", "angry"]):
        return "angry"
    if any(k in lower for k in ["love", "happy", "great", "amazing", "joy", "smile"]):
        return "happy"
    if any(k in lower for k in ["sad", "cry", "sorry", "heartbroken", "pain", "lost"]):
        return "sad"
    if any(k in lower for k in ["wow", "shock", "surprised", "what", "oh no"]):
        return "surprised"
    return "neutral"


def enrich_transcript_with_emotion(folder):
    """Attach emotion metadata to transcript entries for TTS control."""
    trans_path = Path(folder) / "translation.json"
    if not trans_path.exists():
        return

    with open(trans_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    for item in transcript:
        item["emotion"] = detect_emotion_from_text(item.get("translation", item.get("text", "")))

    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)


def ensure_speaker_consistency(folder):
    profiles = build_speaker_profiles(folder)
    for speaker_id, profile in profiles.items():
        if not profile.get("voice_prompt") or not os.path.exists(profile["voice_prompt"]):
            continue
    return profiles
