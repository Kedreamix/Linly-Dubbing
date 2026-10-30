import json
import os
import re
import librosa
import numpy as np
from loguru import logger
from audiostretchy.stretch import stretch_audio

from .utils import save_wav, save_wav_norm
from .step042_tts_xtts import tts as xtts_tts
from .step043_tts_cosyvoice import tts as cosyvoice_tts
from .step044_tts_edge_tts import tts as edge_tts
from .cn_tx import TextNorm

normalizer = TextNorm()


def preprocess_text(text):
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text


def adjust_audio_length(wav_path, desired_length, sample_rate=24000, min_speed_factor=0.6, max_speed_factor=1.1):
    try:
        wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    except Exception:
        if wav_path.endswith('.wav'):
            wav_path = wav_path.replace('.wav', '.mp3')
        wav, sample_rate = librosa.load(wav_path, sr=sample_rate)

    current_length = len(wav) / sample_rate
    speed_factor = max(min(desired_length / current_length, max_speed_factor), min_speed_factor)
    desired_length = current_length * speed_factor

    if wav_path.endswith('.wav'):
        target_path = wav_path.replace('.wav', '_adjusted.wav')
    elif wav_path.endswith('.mp3'):
        target_path = wav_path.replace('.mp3', '_adjusted.wav')

    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    return wav[: int(desired_length * sample_rate)], desired_length


TTS_DEFAULT_METHOD = "cosyvoice"

TTS_SUPPORT = {
    "cosyvoice": ["বাংলা", "English", "中文", "हिन्दी", "日本語", "한국어", "Français", "Español"],
    "xtts": ["English", "中文", "日本語", "한국어", "Français", "Español"],
    "EdgeTTS": ["English", "中文", "日本語", "한국어", "Français", "Español"],
}


def generate_wavs(method, folder, target_language='বাংলা', voice='bn'):
    method = TTS_DEFAULT_METHOD if method is None else str(method).lower()
    if method not in TTS_SUPPORT:
        raise ValueError(f"Unsupported TTS method: {method}")

    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    os.makedirs(output_folder, exist_ok=True)

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    if target_language not in TTS_SUPPORT[method]:
        raise ValueError(f"{method} does not support {target_language}")

    full_wav = np.zeros((0,))
    for i, line in enumerate(transcript):
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{line["speaker"]}.wav') if 'speaker' in line else os.path.join(folder, 'audio_vocals.wav')

        if method == 'cosyvoice':
            cosyvoice_tts(text, output_path, speaker_wav, target_language=target_language)
        elif method == 'xtts':
            xtts_tts(text, output_path, speaker_wav, target_language=target_language)
        else:
            edge_tts(text, output_path, target_language=target_language, voice=voice)

        start = line.get('start', 0)
        end = line.get('end', 0)
        length = max(end - start, 0.1)
        last_end = len(full_wav) / 24000
        if start > last_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000),))))
        start = len(full_wav) / 24000
        line['start'] = start
        if i < len(transcript) - 1:
            next_line = transcript[i + 1]
            next_end = next_line.get('end', start + 1.0)
            end = min(start + length, next_end)
        wav, length = adjust_audio_length(output_path, end - start)
        full_wav = np.concatenate((full_wav, wav))
        line['end'] = start + length

    vocal_wav, sr = librosa.load(os.path.join(folder, 'audio_vocals.wav'), sr=24000)
    if np.max(np.abs(vocal_wav)) > 0:
        full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav))
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))

    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    len_full = len(full_wav)
    len_inst = len(instruments_wav)
    if len_full > len_inst:
        instruments_wav = np.pad(instruments_wav, (0, len_full - len_inst), mode='constant')
    elif len_inst > len_full:
        full_wav = np.pad(full_wav, (0, len_inst - len_full), mode='constant')
    combined_wav = full_wav + instruments_wav
    save_wav_norm(combined_wav, os.path.join(folder, 'audio_combined.wav'))
    return os.path.join(folder, 'audio_combined.wav'), os.path.join(folder, 'audio.wav')


def generate_all_wavs_under_folder(root_folder, method=None, target_language='বাংলা', voice='bn'):
    wav_combined, wav_ori = None, None
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files and 'audio_combined.wav' not in files:
            wav_combined, wav_ori = generate_wavs(method or TTS_DEFAULT_METHOD, root, target_language, voice)
        elif 'audio_combined.wav' in files:
            wav_combined, wav_ori = os.path.join(root, 'audio_combined.wav'), os.path.join(root, 'audio.wav')
    return f'Generated all wavs under {root_folder}', wav_combined, wav_ori
