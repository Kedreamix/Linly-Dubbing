# pip install modelscope
from modelscope import snapshot_download

# https://modelscope.cn/models/AI-ModelScope/XTTS-v2
snapshot_download('AI-ModelScope/XTTS-v2', local_dir='models/TTS/XTTS-v2')

# https://modelscope.cn/models/iic/CosyVoice-300M
# snapshot_download('iic/CosyVoice-300M', local_dir='models/TTS/CosyVoice-300M')

# https://modelscope.cn/models/qwen/qwen1.5-4b-chat
snapshot_download('qwen/Qwen1.5-4B-Chat', local_dir='models/LLM/Qwen1.5-4B-Chat')

# https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat
# snapshot_download('qwen/Qwen1.5-1.8B-Chat', local_dir='models/LLM/Qwen1.5-1.8B-Chat')

# https://modelscope.cn/models/keepitsimple/faster-whisper-large-v3
snapshot_download('keepitsimple/faster-whisper-large-v3', local_dir='models/ASR/whisper/faster-whisper-large-v3')

# 需要申请自动下载
# https://modelscope.cn/models/mirror013/speaker-diarization-3.1
# snapshot_download('mirror013/speaker-diarization-3.1', local_dir='models/ASR/whisper/speaker-diarization-3.1')
