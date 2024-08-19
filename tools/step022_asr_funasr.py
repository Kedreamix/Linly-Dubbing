import json
import time
from funasr import AutoModel
import os
from loguru import logger
import torch
from dotenv import load_dotenv
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

    # 定义模型文件夹路径
    model_path = "models/ASR/FunASR/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_model_path = "models/ASR/FunASR/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_model_path = "models/ASR/FunASR/punc_ct-transformer_cn-en-common-vocab471067-large"
    spk_model_path = "models/ASR/FunASR/speech_campplus_sv_zh-cn_16k-common"
 
    # funasr_model = AutoModel(
    #     model="paraformer-zh", # iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch  
    #     vad_model="fsmn-vad",  # iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
    #     punc_model="ct-punc",  # iic/punc_ct-transformer_cn-en-common-vocab471067-large
    #     spk_model="cam++",     # iic/speech_campplus_sv_zh-cn_16k-common
    # )
    # 加载模型，如果路径存在则使用本地路径，否则使用默认模型
    funasr_model = AutoModel(
        model=model_path if os.path.isdir(model_path) else "paraformer-zh",
        vad_model=vad_model_path if os.path.isdir(vad_model_path) else "fsmn-vad",
        punc_model=punc_model_path if os.path.isdir(punc_model_path) else "ct-punc",
        spk_model=spk_model_path if os.path.isdir(spk_model_path) else "cam++",
    )
    t_end = time.time()
    logger.info(f'Loaded FunASR model in {t_end - t_start:.2f}s')


def funasr_transcribe_audio(wav_path, device='auto', batch_size=1, diarization=True):
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
    # print(rec_result)
    transcript = [{'start': sentence['timestamp'][0][0]/1000, 'end': sentence['timestamp'][-1][-1]/1000, 'text': sentence['text'].strip(), 'speaker': f"SPEAKER_{sentence.get('spk', 0):02d}"} for sentence in rec_result['sentence_info']] 
    return transcript

if __name__ == '__main__':
    for root, dirs, files in os.walk("videos"):
        if 'audio_vocals.wav' in files:
            logger.info(f'Transcribing {os.path.join(root, "audio_vocals.wav")}')
            transcript = funasr_transcribe_audio(os.path.join(root, "audio_vocals.wav"))
            print(transcript)
            break