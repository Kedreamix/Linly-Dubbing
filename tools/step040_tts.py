import json
import os
import re
import librosa

from loguru import logger
import numpy as np

from .utils import save_wav, save_wav_norm
# from .step041_tts_bytedance import tts as bytedance_tts
from .step042_tts_xtts import tts as xtts_tts
from .step043_tts_cosyvoice import tts as cosyvoice_tts
from .step044_tts_edge_tts import tts as edge_tts
from .cn_tx import TextNorm
from audiostretchy.stretch import stretch_audio
normalizer = TextNorm()
def preprocess_text(text):
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    # 使用正则表达式在字母和数字之间插入空格
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text
    
    
def adjust_audio_length(wav_path, desired_length, sample_rate = 24000, min_speed_factor = 0.6, max_speed_factor = 1.1):
    wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    current_length = len(wav)/sample_rate
    speed_factor = max(
        min(desired_length / current_length, max_speed_factor), min_speed_factor)
    desired_length = current_length * speed_factor
    target_path = wav_path.replace('.wav', f'_adjusted.wav')
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    return wav[:int(desired_length*sample_rate)], desired_length

tts_support_languages = {
    # XTTS-v2 supports 17 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi).
    'xtts': ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish'],
    'bytedance': [],
    'GPTSoVits': [],
    'EdgeTTS': ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish'],
    # zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    'cozyvoice': ['中文', '粤语', 'English', 'Japanese', 'Korean', 'French'], 
}

def generate_wavs(method, folder, target_language='中文', voice = 'zh-CN-XiaoxiaoNeural'):
    assert method in ['xtts', 'bytedance', 'cosyvoice', 'EdgeTTS']
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    speakers = set()
    
    for line in transcript:
        speakers.add(line['speaker'])
    num_speakers = len(speakers)
    logger.info(f'Found {num_speakers} speakers')

    if target_language not in tts_support_languages[method]:
        logger.error(f'{method} does not support {target_language}')
        return f'{method} does not support {target_language}'
        
    full_wav = np.zeros((0, ))
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        # if num_speakers == 1:
            # bytedance_tts(text, output_path, speaker_wav, voice_type='BV701_streaming')
        
        if method == 'bytedance':
            bytedance_tts(text, output_path, speaker_wav, target_language = target_language)
        elif method == 'xtts':
            xtts_tts(text, output_path, speaker_wav, target_language = target_language)
        elif method == 'cosyvoice':
            cosyvoice_tts(text, output_path, speaker_wav, target_language = target_language)
        elif method == 'EdgeTTS':
            edge_tts(text, output_path, target_language = target_language, voice = voice)
        start = line['start']
        end = line['end']
        length = end-start
        last_end = len(full_wav)/24000
        if start > last_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000), ))))
        start = len(full_wav)/24000
        line['start'] = start
        if i < len(transcript) - 1:
            next_line = transcript[i+1]
            next_end = next_line['end']
            end = min(start + length, next_end)
        wav, length = adjust_audio_length(output_path, end-start)

        full_wav = np.concatenate((full_wav, wav))
        line['end'] = start + length
        
    vocal_wav, sr = librosa.load(os.path.join(folder, 'audio_vocals.wav'), sr=24000)
    full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav))
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    
    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)
    
    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')
    combined_wav = full_wav + instruments_wav
    # combined_wav /= np.max(np.abs(combined_wav))
    save_wav_norm(combined_wav, os.path.join(folder, 'audio_combined.wav'))
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')
    return os.path.join(folder, 'audio_combined.wav'), os.path.join(folder, 'audio.wav')

def generate_all_wavs_under_folder(root_folder, method, target_language='中文', voice = 'zh-CN-XiaoxiaoNeural'):
    wav_combined, wav_ori = None, None
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files and 'audio_combined.wav' not in files:
            wav_combined, wav_ori = generate_wavs(method, root, target_language, voice)
        elif 'audio_combined.wav' in files:
            wav_combined, wav_ori = os.path.join(root, 'audio_combined.wav'), os.path.join(root, 'audio.wav')
            logger.info(f'Wavs already generated in {root}')
    return f'Generated all wavs under {root_folder}', wav_combined, wav_ori

if __name__ == '__main__':
    folder = r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候'
    generate_wavs('xtts', folder)
