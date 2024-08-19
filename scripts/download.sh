# 下载 wav2vec2 模型并保存到指定路径，如果文件已经存在，则跳过下载
mkdir -p models/ASR/whisper & wget -nc https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth \
    -O models/ASR/whisper/wav2vec2_fairseq_base_ls960_asr_ls960.pth

# 执行下载脚本
python scripts/modelscope_download.py