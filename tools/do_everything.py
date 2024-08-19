import json
import os
import time
from loguru import logger
from .step000_video_downloader import get_info_list_from_url, download_single_video, get_target_folder
from .step010_demucs_vr import separate_all_audio_under_folder, init_demucs
from .step020_asr import transcribe_all_audio_under_folder
from .step021_asr_whisperx import init_whisperx, init_diarize
from .step022_asr_funasr import init_funasr
from .step030_translation import translate_all_transcript_under_folder
from .step040_tts import generate_all_wavs_under_folder
from .step042_tts_xtts import init_TTS
from .step043_tts_cosyvoice import init_cosyvoice
from .step050_synthesize_video import synthesize_all_video_under_folder
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_video(info, root_folder, resolution,
                  demucs_model, device, shifts, 
                  asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers, 
                  translation_method, translation_target_language, 
                  tts_method, tts_target_language, voice,
                  subtitles, speed_up, fps, background_music, bgm_volume, video_volume, 
                  target_resolution, max_retries):
    local_time = time.localtime()

    for retry in range(max_retries):
        try:
            if isinstance(info, str) and info.endswith('.mp4'):
                folder = os.path.dirname(info)
                # os.rename(info, os.path.join(folder, 'download.mp4'))
            else:
                folder = get_target_folder(info, root_folder)
                if folder is None:
                    logger.warning(f'Failed to get target folder for video {info["title"]}')
                    return False

                folder = download_single_video(info, root_folder, resolution)
                if folder is None:
                    logger.warning(f'Failed to download video {info["title"]}')
                    return True

            logger.info(f'Process video in {folder}')
            separate_all_audio_under_folder(
                folder, model_name=demucs_model, device=device, progress=True, shifts=shifts)
            transcribe_all_audio_under_folder(
                folder, asr_method=asr_method, whisper_model_name=whisper_model, device=device, 
                batch_size=batch_size, diarization=diarization, 
                min_speakers=whisper_min_speakers,
                max_speakers=whisper_max_speakers)
            
            translate_all_transcript_under_folder(
                folder, method = translation_method, target_language=translation_target_language
            )
            generate_all_wavs_under_folder(folder, method = tts_method, target_language=tts_target_language, voice=voice)
            _, ouput_video = synthesize_all_video_under_folder(folder, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=target_resolution, 
                                                                background_music=background_music, bgm_volume=bgm_volume, video_volume=video_volume)
            return True, ouput_video
        except Exception as e:
            logger.error(f'Error processing video {info["title"]}: {e}')
    return False, None


def do_everything(root_folder, url, num_videos=5, resolution='1080p',
                  demucs_model='htdemucs_ft', device='auto', shifts=5, 
                  asr_method='WhisperX', whisper_model='large', batch_size=32, diarization=False, whisper_min_speakers=None, whisper_max_speakers=None, 
                  translation_method = 'LLM',translation_target_language='简体中文', 
                  tts_method='xtts', tts_target_language='中文', voice='zh-CN-XiaoxiaoNeural',
                  subtitles=True, speed_up=1.00, fps=30, 
                  background_music=None, bgm_volume=0.5, video_volume=1.0, target_resolution='1080p', 
                  max_workers=3, max_retries=5):
    success_list = []
    fail_list = []

    url = url.replace(' ', '').replace('，', '\n').replace(',', '\n')
    urls = [_ for _ in url.split('\n') if _]
    
    # 使用线程池执行任务
    with ThreadPoolExecutor() as executor:
        # Submitting the tasks
        # video_info_future = executor.submit(get_info_list_from_url, urls, num_videos)
        executor.submit(init_demucs)
        if tts_method == 'xtts':
            executor.submit(init_TTS)
        elif tts_method == 'cosyvoice':
            executor.submit(init_cosyvoice)
        if asr_method == 'WhisperX':
            executor.submit(init_whisperx)
            if diarization:
                executor.submit(init_diarize)
        elif asr_method == 'FunASR':
            executor.submit(init_funasr)
        # Waiting for the get_info_list_from_url task to complete and storing its result
        # video_info_list = video_info_future.result()
    out_video = None
    if url.endswith('.mp4'):
        import shutil
        # 获取原始视频文件名（不带路径）
        original_file_name = os.path.basename(url)
        
        # 去除文件扩展名，生成文件夹名称
        new_folder_name = os.path.splitext(original_file_name)[0]
        
        # 构建新文件夹的完整路径
        new_folder_path = os.path.join(root_folder, new_folder_name)
        
        # 在 root_folder 下创建该文件夹
        os.makedirs(new_folder_path, exist_ok=True)
        
        # 构建原始文件的完整路径
        original_file_path = os.path.join(root_folder, original_file_name)
        
        # 构建新位置的完整路径
        new_file_path = os.path.join(new_folder_path, "download.mp4")
        
        # 将视频文件移动到新创建的文件夹中并重命名
        shutil.copy(original_file_path, new_file_path)
        # 在 root_folder 下创建该文件夹
        os.makedirs(new_folder_path, exist_ok=True)

        success, output_video = process_video(new_file_path, root_folder, resolution, 
                                    demucs_model, device, shifts, 
                                    asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers, 
                                    translation_method, translation_target_language, 
                                    tts_method, tts_target_language, voice,
                                    subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                                    target_resolution, max_retries)
        if success:
            return 'Success', out_video
        else:
            return 'Fail', None
    else:
        for info in get_info_list_from_url(urls, num_videos):
            success, output_video = process_video(info, root_folder, resolution, 
                                    demucs_model, device, shifts, 
                                    asr_method, whisper_model, batch_size, diarization, whisper_min_speakers, whisper_max_speakers, 
                                    translation_method, translation_target_language, 
                                    tts_method, tts_target_language, voice,
                                    subtitles, speed_up, fps, background_music, bgm_volume, video_volume,
                                    target_resolution, max_retries)
            if success:
                success_list.append(info)
                out_video = output_video
            else:
                fail_list.append(info)

    return f'Success: {len(success_list)}\nFail: {len(fail_list)}', out_video


if __name__ == '__main__':
    do_everything(
        root_folder = 'videos', 
        url = 'https://www.bilibili.com/video/BV1kr421M7vz/',
        translation_method = 'LLM', 
        # translation_method = 'Google Translate', translation_target_language = '简体中文',
        )