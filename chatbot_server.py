# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse, JSONResponse
# from pydantic import BaseModel
# from threading import Thread
# from sse_starlette.sse import EventSourceResponse
# import logging

# # ------- 配置 -------
# MODEL_PATH = "Qwen/Qwen3-0.6B"  # 替换为你的本地模型路径
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# # ------- 加载模型 -------
# print("Loading model...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=TORCH_DTYPE,
#     device_map="auto",
#     trust_remote_code=True
# )
# model.eval()
# print(f"Model loaded on {DEVICE}")

# # ------- FastAPI 应用 -------
# app = FastAPI(title="Qwen3-4B Streaming Server")

# class ChatRequest(BaseModel):
#     prompt: str
#     max_new_tokens: int = 512
#     temperature: float = 0.7
#     top_p: float = 0.8
#     stream: bool = False

# def generate_stream(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
#     inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
#     streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     generation_kwargs = dict(
#         **inputs,
#         streamer=streamer,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=temperature,
#         top_p=top_p,
#     )

#     # 在子线程中启动生成
#     thread = Thread(target=model.generate, kwargs=generation_kwargs)
#     thread.start()

#     for new_text in streamer:
#         if new_text:
#             yield new_text

# @app.post("/v1/chat/completions")
# async def chat_completions(request: ChatRequest):
#     if request.stream:
#         return EventSourceResponse(
#             generate_stream(
#                 request.prompt,
#                 request.max_new_tokens,
#                 request.temperature,
#                 request.top_p
#             ),
#             media_type="text/event-stream"
#         )
#     else:
#         # 非流式：一次性生成
#         inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=request.max_new_tokens,
#                 do_sample=True,
#                 temperature=request.temperature,
#                 top_p=request.top_p,
#             )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # 若 prompt 被包含在 response 中，可切掉（Qwen 通常不会重复）
#         return JSONResponse({"response": response})
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ← 新增：CORS 支持
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from threading import Thread
from sse_starlette.sse import EventSourceResponse
import logging

# ------- 配置 -------
# uvicorn server:app --host 0.0.0.0 --port 8000 --reload
MODEL_PATH = "Qwen/Qwen3-4B" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ------- 加载模型 -------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_DTYPE,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print(f"Model loaded on {DEVICE}")

# ------- FastAPI 应用 -------
app = FastAPI(title="Qwen3-4B Streaming Server")

# ←←← 新增：CORS 中间件配置 →→→
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境建议指定具体域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = False

def generate_stream(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        if new_text:
            yield new_text

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.stream:
        return EventSourceResponse(
            generate_stream(
                request.prompt,
                request.max_new_tokens,
                request.temperature,
                request.top_p
            ),
            media_type="text/event-stream"
        )
    else:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return JSONResponse({"response": response})


# uvicorn chatbot_server:app --host 0.0.0.0 --port 5000 --reload