
import os
import torch
import numpy as np
from dotenv import load_dotenv
from .step021_asr_whisperx import whisperx_transcribe_audio
from .step022_asr_funasr import funasr_transcribe_audio
from .utils import save_wav
import json
import librosa
from loguru import logger
load_dotenv()

def merge_segments(transcript, ending='!"\').:;?]}~！“”’）。：；？】'):
    merged_transcription = []
    buffer_segment = None

    for segment in transcript:
        if buffer_segment is None:
            buffer_segment = segment
        else:
            # Check if the last character of the 'text' field is a punctuation mark
            if buffer_segment['text'][-1] in ending:
                # If it is, add the buffered segment to the merged transcription
                merged_transcription.append(buffer_segment)
                buffer_segment = segment
            else:
                # If it's not, merge this segment with the buffered segment
                buffer_segment['text'] += ' ' + segment['text']
                buffer_segment['end'] = segment['end']

    # Don't forget to add the last buffered segment
    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription

def generate_speaker_audio(folder, transcript):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict = dict()
    length = len(audio_data)
    delay = 0.05
    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end']+delay) * samplerate), length)
        speaker_segment_audio = audio_data[start:end]
        speaker_dict[segment['speaker']] = np.concatenate((speaker_dict.get(
            segment['speaker'], np.zeros((0, ))), speaker_segment_audio))

    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)
    
    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(
            speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path)


def transcribe_audio(method, folder, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True,min_speakers=None, max_speakers=None):
    if os.path.exists(os.path.join(folder, 'transcript.json')):
        logger.info(f'Transcript already exists in {folder}')
        return True
    
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        return False
    
    logger.info(f'Transcribing {wav_path}')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if method == 'WhisperX':
        transcript = whisperx_transcribe_audio(wav_path, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers)
    elif method == 'FunASR':
        transcript = funasr_transcribe_audio(wav_path, device, batch_size, diarization)
    else:
        logger.error('Invalid ASR method')
        raise ValueError('Invalid ASR method')

    transcript = merge_segments(transcript)
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
    generate_speaker_audio(folder, transcript)
    return transcript

def transcribe_all_audio_under_folder(folder, asr_method, whisper_model_name: str = 'large', device='auto', batch_size=32, diarization=False, min_speakers=None, max_speakers=None):
    transcribe_json = None
    for root, dirs, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            transcribe_json = transcribe_audio(asr_method, root, whisper_model_name, 'models/ASR/whisper', device, batch_size, diarization, min_speakers, max_speakers)
        elif 'transcript.json' in files:
            transcribe_json = json.load(open(os.path.join(root, 'transcript.json'), 'r', encoding='utf-8'))

            # logger.info(f'Transcript already exists in {root}')
    return f'Transcribed all audio under {folder}', transcribe_json

if __name__ == '__main__':
    _, transcribe_json = transcribe_all_audio_under_folder('videos', 'WhisperX')
    print(transcribe_json)
    # _, transcribe_json = transcribe_all_audio_under_folder('videos', 'FunASR')    
    # print(transcribe_json)