# -*- coding: utf-8 -*-
import json
import os
from googletrans import Translator
from dotenv import load_dotenv
import time
from loguru import logger

load_dotenv()
translator = Translator()

def get_necessary_info(info: dict):
    return {
        'title': info['title'],
        'uploader': info['uploader'],
        'description': info['description'],
        'upload_date': info['upload_date'],
        'tags': info['tags'],
    }

def ensure_transcript_length(transcript, max_length=4000):
    mid = len(transcript) // 2
    before, after = transcript[:mid], transcript[mid:]
    length = max_length // 2
    return before[:length] + after[-length:]

def summarize(info, transcript, target_language='zh-CN'):
    transcript_text = ' '.join(line['text'] for line in transcript)
    transcript_text = ensure_transcript_length(transcript_text, max_length=2000)
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}".'
    
    full_description = f'{info_message}\n{transcript_text}\n{info_message}\n'
    
    translation = translator.translate(full_description, dest=target_language).text
    logger.info(f'原文: {full_description}')
    logger.info(f'译文: {translation}')
    
    summary = {
        'title': translator.translate(info["title"], dest=target_language).text,
        'summary': translation
    }

    return summary

def translate(folder, target_language='zh-CN'):
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True
    
    info_path = os.path.join(folder, 'download.info.json')
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
        summary = summarize(info, transcript, target_language)
        if summary is None:
            logger.error(f'Failed to summarize {folder}')
            return False
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    translation_path = os.path.join(folder, 'translation.json')
    translation = [translator.translate(line['text'], dest=target_language).text for line in transcript]
    for i, line in enumerate(transcript):
        line['translation'] = translation[i]
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return True

def translate_all_transcript_under_folder(folder, target_language):
    for root, dirs, files in os.walk(folder):
        if 'transcript.json' in files and 'translation.json' not in files:
            translate(root, target_language)
    return f'Translated all videos under {folder}'

if __name__ == '__main__':
    translate_all_transcript_under_folder(
        r'videos/黑纹白斑马/20240805 中配学习Python通过10个项目免费15小时课程 - Indently', 'en')
