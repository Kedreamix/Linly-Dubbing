
import os
from dotenv import load_dotenv
from .step021_asr_whisperx import transcribe_audio as whisperx_transcribe_audio
from .step022_asr_funasr import transcribe_audio as funasr_transcribe_audio
from .utils import save_wav
import json
load_dotenv()

            

def transcribe_all_audio_under_folder(folder, asr_method, whisper_model_name: str = 'large', device='auto', batch_size=32, diarization=False, min_speakers=None, max_speakers=None):
    transcribe_json = None
    for root, dirs, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            if asr_method == 'WhisperX':
                transcribe_json = whisperx_transcribe_audio(root, whisper_model_name, 'models/ASR/whisper', device, batch_size, diarization, min_speakers, max_speakers)
            elif asr_method == 'FunASR':
                transcribe_json =  funasr_transcribe_audio(root, device, batch_size, diarization)
        elif 'transcript.json' in files:
            transcribe_json = json.load(open(os.path.join(root, 'transcript.json'), 'r', encoding='utf-8'))

            # logger.info(f'Transcript already exists in {root}')
    return f'Transcribed all audio under {folder}', transcribe_json

if __name__ == '__main__':
    transcribe_all_audio_under_folder('videos', 'WhisperX',)
    # transcribe_all_audio_under_folder('videos', 'FunASR')    
    