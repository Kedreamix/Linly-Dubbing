import os
from TTS.api import TTS
from loguru import logger
import numpy as np
import torch
import time
from .utils import save_wav
model = None

'''
Supported languages: Arabic: ar, Brazilian Portuguese: pt , Mandarin Chinese: zh-cn, Czech: cs, Dutch: nl, English: en, French: fr, German: de, Italian: it, Polish: pl, Russian: ru, Spanish: es, Turkish: tr, Japanese: ja, Korean: ko, Hungarian: hu, Hindi: hi
'''
def init_TTS():
    load_model()
    
def load_model(model_path="models/TTS/XTTS-v2", device='auto'):
    global model
    if model is not None:
        return

    if device=='auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          
    logger.info(f'Loading TTS model from {model_path}')
    t_start = time.time()
    if os.path.isdir(model_path):
        print(f"Loading TTS model from {model_path}")
        model = TTS(
            model_path = model_path,
            config_path = os.path.join(model_path, 'config.json'),
        ).to(device)
    else:
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    t_end = time.time()
    logger.info(f'TTS model loaded in {t_end - t_start:.2f}s')

# XTTS-v2 supports 17 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), 
# Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), 
# Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi).
language_map = {
    '中文': 'zh-cn',
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Polish': 'pl',
    'Turkish': 'tr',
    'Russian': 'ru',
    'Dutch': 'nl',
    'Czech': 'cs',
    'Arabic': 'ar',
    'Hungarian': 'hu',
    'Hindi': 'hi',
    'Korean': 'ko',
}
def tts(text, output_path, speaker_wav, model_name="models/TTS/XTTS-v2", device='auto', target_language='中文'):
    global model
    language = language_map[target_language]
    assert language in ['ar', 'pt', 'zh-cn', 'cs', 'nl', 'en', 'fr', 'de', 'it', 'pl', 'ru', 'es', 'tr', 'ja', 'ko', 'hu', 'hi']
    if os.path.exists(output_path):
        logger.info(f'TTS {text} 已存在')
        return
    
    if model is None:
        load_model(model_name, device)
    
    for retry in range(3):
        try:
            wav = model.tts(text, speaker_wav=speaker_wav, language=language)
            wav = np.array(wav)
            save_wav(wav, output_path)
            logger.info(f'TTS {text}')
            break
        except Exception as e:
            logger.warning(f'TTS {text} 失败')
            logger.warning(e)


if __name__ == '__main__':
    speaker_wav = r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候/audio_vocals.wav'
    os.makedirs('playground', exist_ok=True)
    while True:
        text = input('请输入：')
        tts(text, f'playground/{text}.wav', speaker_wav)
