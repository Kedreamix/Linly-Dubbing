# pip install huggingface_hub
from huggingface_hub import snapshot_download

# https://huggingface.co/coqui/XTTS-v2
snapshot_download('coqui/XTTS-v2', local_dir='models/TTS/XTTS-v2', resume_download=True, local_dir_use_symlinks=False)

# https://huggingface.co/FunAudioLLM/CosyVoice-300M
# snapshot_download('FunAudioLLM/CosyVoice-300M', local_dir='models/TTS/CosyVoice-300M', resume_download=True, local_dir_use_symlinks=False)

# https://huggingface.co/Qwen/Qwen1.5-4B-Chat
snapshot_download('Qwen/Qwen1.5-4B-Chat', local_dir='models/LLM/Qwen1.5-4B-Chat', resume_download=True, local_dir_use_symlinks=False)

# https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat
snapshot_download('Qwen/Qwen1.5-1.8B-Chat', local_dir='models/LLM/Qwen1.5-1.8B-Chat', resume_download=True,  local_dir_use_symlinks=False)

# https://huggingface.co/Systran/faster-whisper-large-v3
snapshot_download('Systran/faster-whisper-large-v3', local_dir='models/ASR/whisper/faster-whisper-large-v3', resume_download=True, local_dir_use_symlinks=False)

# 需要申请自动下载
# https://huggingface.co/pyannote/speaker-diarization-3.1
# snapshot_download('pyannote/speaker-diarization-3.1', local_dir='models/ASR/whisper/speaker-diarization-3.1', resume_download=True, local_dir_use_symlinks=False)
