# -*- coding: utf-8 -*-
import json
import os
import re

from dotenv import load_dotenv
import time
from loguru import logger
from .step031_translation_openai import summarize as openai_summarize, _translate as openai_translate
from .step032_translation_llm import summarize as llm_summarize, _translate as llm_translate
# from .step033_translation_google import summarize as google_translate_summarize, translate as google_translate_translate

load_dotenv()

def get_necessary_info(info: dict):
    return {
        'title': info['title'],
        'uploader': info['uploader'],
        'description': info['description'],
        'upload_date': info['upload_date'],
        # 'categories': info['categories'],
        'tags': info['tags'],
    }


def ensure_transcript_length(transcript, max_length=4000):
    mid = len(transcript)//2
    before, after = transcript[:mid], transcript[mid:]
    length = max_length//2
    return before[:length] + after[-length:]

def split_text_into_sentences(para):
    para = re.sub('([。！？\?])([^，。！？\?”’》])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def split_sentences(translation):
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item['translation']
        sentences = split_text_into_sentences(translation_text)
        duration_per_char = (item['end'] - item['start']
                             ) / len(translation_text)
        sentence_start = 0
        for sentence in sentences:
            sentence_end = start + duration_per_char * len(sentence)

            # Append the new item
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": text,
                "speaker": speaker,
                "translation": sentence
            })

            # Update the start for the next sentence
            start = sentence_end
            sentence_start += len(sentence)
    return output_data

def translate(method, folder, target_language='简体中文'):
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True
    
    info_path = os.path.join(folder, 'download.info.json')
    # 不一定要download.info.json
    if not os.path.exists(info_path):
        return False
    
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    info = get_necessary_info(info)
    
    transcript_path = os.path.join(folder, 'transcript.json')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path, 'r', encoding='utf-8'))
    else:
        if method == 'OpenAI':
            summary = openai_summarize(info, transcript, target_language)
        elif method == 'Qwen':
            summary = llm_summarize(info, transcript, target_language)
        elif method == 'Google Translate':
            summary = google_translate_summarize(info, transcript, target_language)

        if summary is None:
            logger.error(f'Failed to summarize {folder}')
            return False
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    translation_path = os.path.join(folder, 'translation.json')
    if method == 'OpenAI':
        translation = openai_translate(summary, transcript, target_language)
    elif method == 'Qwen':
        translation = llm_translate(summary, transcript, target_language)
    elif method == 'Google Translate':
        translation = google_translate_translate(summary, transcript, target_language)
    for i, line in enumerate(transcript):
        line['translation'] = translation[i]
    transcript = split_sentences(transcript)
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return summary, transcript

def translate_all_transcript_under_folder(folder, method, target_language):
    summary_json , translate_json = None, None
    for root, dirs, files in os.walk(folder):
        if 'transcript.json' in files and 'translation.json' not in files:
            summary_json , translate_json = translate(method, root, target_language)
        elif 'translation.json' in files:
            summary_json = json.load(open(os.path.join(root, 'summary.json'), 'r', encoding='utf-8'))
            translate_json = json.load(open(os.path.join(root, 'translation.json'), 'r', encoding='utf-8'))
    print(summary_json, translate_json)
    return f'Translated all videos under {folder}',summary_json , translate_json

if __name__ == '__main__':
    translate_all_transcript_under_folder(
        r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候',
          'Qwen' , '简体中文')
    # translate_all_transcript_under_folder('OpenAI' ,
    #                                       r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候', 
    #                                       '简体中文')