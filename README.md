## Qwen Local Deployment Example

### Introduction
> This is an example project for demonstrating how to deploy the Qwen model locally. It includes the necessary code and documentation to help users quickly get started and run the Qwen model locally.


### Directory Structure
```plaintext
. /
├── README.md          # Project description file
├── requirements.txt   # List of dependencies
├── chatbot_server.py # Qwen model server code, encapsulating an interface that supports streaming processing to return generation results in real-time and provide them to the front end
├── chatbot_client.py # Qwen model client code, demonstrating how to call the server interface for dialogue, using streamlit to build a simple front-end interface
└── audio_server.py   # Audio processing server code, encapsulating an interface for audio-to-text conversion to support voice input and return the converted text to the front end
```


## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

1. Start the audio processing server (optional, if voice input is needed):
```bash
   uvicorn audio_server:app --host 0.0.0.0 --port 5001 --reload
   ```
2. Start the Qwen model server:
```bash
   uvicorn chatbot_server:app --host 0.0.0.0 --port 5000 --reload
   ```
3. Start the front-end interface:
```bash
   streamlit run chatbot_client.py
   ```
4. Open your browser and go to `http://localhost:8501` to use the chat interface.

## Note 
> The Qwen-4b model must use GPU deployment. Please ensure that your local machine has a compatible GPU and the necessary CUDA drivers installed.

## Development Environment
- Python 3.8+
- Streamlit
- FastAPI
- Uvicorn
- Qwen Model SDK
- RTX5060Ti GPU with 16GB VRAM

## License
This project is licensed under the MIT License. See the LICENSE file for details.
