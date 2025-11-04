import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "kk90ujhun/whisper-small-zh"
# --- 配置 ---
audio_path = "audio-data/zh-arms_0010.wav"  # 替换为你的本地 .mp4 文件路径
# model_name = "ryL\whisper-small_zh-TW"

# --- 加载模型和处理器 ---
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None  # 自动检测语言 & 任务

# --- 从本地 .mp4 加载音频 ---
# librosa 可直接读取 .mp4（需安装 ffmpeg）
audio_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)

# 注意：Whisper 要求输入是 16000 Hz 的单声道 float32 numpy 数组
# librosa.load(sr=16000) 已完成重采样和转单声道

# --- 特征提取 ---
input_features = processor(
    audio_array, 
    sampling_rate=16000, 
    return_tensors="pt"
).input_features

# --- 推理 ---
with torch.no_grad():
    predicted_ids = model.generate(input_features)

# --- 解码 ---
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("Transcription:", transcription)


# lorenzoncina/whisper-medium-zh

# huggingface-cli download --resume-download lorenzoncina/whisper-medium-zh --local-dir lorenzoncina/whisper-medium-zh