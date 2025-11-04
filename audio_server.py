# speech_to_text_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = FastAPI(title="语音转文字服务")

# CORS（可选）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型（启动时加载一次）
print("正在加载 Whisper 模型...")
processor = WhisperProcessor.from_pretrained("lorenzoncina/whisper-medium-zh")
model = WhisperForConditionalGeneration.from_pretrained("lorenzoncina/whisper-medium-zh")
model.config.forced_decoder_ids = None
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.mp4', '.flac')):
        raise HTTPException(status_code=400, detail="仅支持音频文件")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 加载音频
        audio_array, _ = librosa.load(tmp_path, sr=16000, mono=True)
        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"text": text.strip()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
    finally:
        os.unlink(tmp_path)  # 删除临时文件

# 启动命令: uvicorn audio_server:app --host 0.0.0.0 --port 5001 --reload