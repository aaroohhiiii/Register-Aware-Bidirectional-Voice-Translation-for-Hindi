FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch (~200 MB instead of ~2 GB for the CUDA build)
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-render.txt .
RUN pip install -r requirements-render.txt

# Bake models into the image so the server starts without downloading them
RUN python -c "import stanza; stanza.download('hi', model_dir='/app/.cache/stanza', processors={'tokenize': 'hdtb', 'pos': 'hdtb_charlm'})" \
 && python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration as W; WhisperProcessor.from_pretrained('vasista22/whisper-hindi-small'); W.from_pretrained('vasista22/whisper-hindi-small')" \
 && python -c "from transformers import AutoTokenizer, VitsModel; AutoTokenizer.from_pretrained('facebook/mms-tts-hin'); VitsModel.from_pretrained('facebook/mms-tts-hin')"

COPY . .

# Render sets $PORT; app.py binds 0.0.0.0:$PORT when it is present
EXPOSE 10000
CMD ["python", "app.py"]
