import json
import time
import librosa
import numpy as np
from funasr import AutoModel
import os
from loguru import logger
import torch
from dotenv import load_dotenv

from tools.utils import save_wav
load_dotenv()

funasr_model = None

def init_funasr():
    load_funasr_model()

    
def load_funasr_model(device='auto'):
    global funasr_model
    if funasr_model is not None:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Loading FunASR model')
    t_start = time.time()
    funasr_model = AutoModel(
        model="paraformer-zh",  
        vad_model="fsmn-vad", 
        punc_model="ct-punc", 
        spk_model="cam++"
    )
    t_end = time.time()
    logger.info(f'Loaded FunASR model in {t_end - t_start:.2f}s')


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

def transcribe_audio(folder,  device='auto', batch_size=1, diarization=True):
    if os.path.exists(os.path.join(folder, 'transcript.json')):
        logger.info(f'Transcript already exists in {folder}')
        return True
    
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        return False
    
    logger.info(f'Transcribing {wav_path}')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_funasr_model(device)
    rec_result = funasr_model.generate(
        wav_path,
        device=device, 
        # batch_size=batch_size,
        return_spk_res=True if diarization else False,
        sentence_timestamp=True,
        return_raw_text=True,
        is_final=True,
        batch_size_s=300
        )[0]
    print(rec_result)
    transcript = [{'start': sentence['timestamp'][0][0]/1000, 'end': sentence['timestamp'][-1][-1]/1000, 'text': sentence['text'].strip(), 'speaker': f"SPEAKER_{sentence.get('spk', 0):02d}"} for sentence in rec_result['sentence_info']]
    # transcript = [{'start': segement['start'], 'end': segement['end'], 'text': segement['text'].strip(), 'speaker': segement.get('speaker', 'SPEAKER_00')} for segement in rec_result['segments']]
    transcript = merge_segments(transcript)
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
    logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
    generate_speaker_audio(folder, transcript)
    return transcript

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
            

def transcribe_all_audio_under_folder(folder, device='auto', batch_size=32):
    for root, dirs, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            transcribe_audio(root, device, batch_size)
    return f'Transcribed all audio under {folder}'

if __name__ == '__main__':
    transcribe_all_audio_under_folder('videos')
    