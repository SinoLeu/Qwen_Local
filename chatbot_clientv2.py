
# pip install audio-recorder-streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple
# chatbot_client.py
import streamlit as st
import requests
import json
import time
from html import escape
# from html import escape
from audio_recorder_streamlit import audio_recorder

# 后端地址（请根据实际部署修改）
BACKEND_URL = "http://172.20.10.3:5000/v1/chat/completions"
TRANSCRIBE_URL = "http://172.20.10.3:5001/transcribe"



# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是 Qwen3-Max，通义千问系列大模型，可以回答问题、创作文字、编程等。有什么我可以帮你的吗？"}
    ]

st.set_page_config(page_title="安普机器人-聊天助手", page_icon="💬")
st.header("💬 安普机器人-聊天助手", divider="rainbow")

# 侧边栏：参数设置 + 清除按钮
with st.sidebar:
    st.subheader("⚙️ 设置")
    max_tokens = st.selectbox("最长回复长度", [512, 1024, 2048, 4096], index=2)
    
    if st.button("🗑️ 清除对话", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "对话已清空。你可以继续提问或说话。"}
        ]
        st.rerun()

# 最长回复长度选择
# max_tokens = st.sidebar.selectbox(
#     "最长回复长度",
#     options=[512, 1024, 2048, 4096],
#     index=2  # 默认 2048
# )


# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # 支持 Markdown 渲染（Streamlit 自动处理安全性）
            st.markdown(msg["content"])
        else:
            st.markdown(escape(msg["content"]))  # 用户输入无需 Markdown

# ========= 新增：语音输入区域 =========
st.markdown("### 🎤 语音输入")
audio_bytes = audio_recorder(
    text="点击麦克风开始录音",
    recording_color="#e74c3c",
    neutral_color="#3498db",
    icon_size="2x",
    key="audio_recorder"
)

prompt = st.chat_input("输入你的问题...")

if audio_bytes is not None:
    # 显示录音回放
    st.audio(audio_bytes, format="audio/wav")

    # 发送音频到转录服务
    files = {
        'file': ('recording.wav', audio_bytes, 'audio/wav')
    }
    try:
        with st.spinner("正在转录音频..."):
            response = requests.post(TRANSCRIBE_URL, files=files, timeout=30)
            response.raise_for_status()
            result = response.json()
            # print("Transcription result:", result)
            transcription = result.get("text", "")
            # print("Transcription:", transcription)
        if transcription:
            st.success("转录完成！")
            st.markdown(f"**转录文本：** {escape(transcription)}")
            # 将transcription 添加到prompt中
            prompt = transcription
            
            # 将转录文本作为用户消息添加到对话中
            st.session_state.messages.append({"role": "user", "content": transcription})
            # st.experimental_rerun()
        else:
            st.error("未能获取转录文本。")
    except Exception as e:
        st.error(f"转录失败: {str(e)}")

# 用户输入
if prompt:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(escape(prompt))

    # 构造 Qwen3 多轮对话 prompt
    full_prompt = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            full_prompt += f"<|im_start|>user\n{msg['content']}\n<|im_end|>\n"
        else:
            full_prompt += f"<|im_start|>assistant\n{msg['content']}\n<|im_end|>\n"
    full_prompt += "<|im_start|>assistant\n"

    # 发送请求
    payload = {
        "prompt": full_prompt + " /no_think",
        "max_new_tokens": max_tokens,
        "top_p": 0.7,
        "stream": True
    }
    start_time = time.time()
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        token_speed_placeholder = st.empty()
        time_placeholder = st.empty()
        full_response = ""
        try:
            with requests.post(BACKEND_URL, json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        # 忽略 <think>...</think> 相关内容
                        if "<think>" in decoded or "</think>" in decoded:
                            continue
                        # 提取 data: 后的内容（类似 SSE）
                        if decoded.startswith("data: "):
                            content = decoded[6:]
                            if content:
                                full_response += content
                                # 实时渲染为 Markdown
                                message_placeholder.markdown(full_response + "▌")
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    token_speed_placeholder.write(f"平均 token 速率: {len(full_response.split()) / elapsed:.2f}/s")
                                    time_placeholder.write(f"耗时: {elapsed:.2f}s")
                                
            # 最终渲染（无光标）
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            error_msg = f"❌ 请求失败: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})