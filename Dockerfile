# Hugging Face Spaces (Docker SDK) image for the Gradio app.
# HF runs the container as uid 1000 and routes traffic to port 7860.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/app/.cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860

WORKDIR /home/user/app

# CPU-only PyTorch (~200 MB instead of ~2 GB for the CUDA build)
RUN pip install --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements-deploy.txt .
RUN pip install --user -r requirements-deploy.txt

# Bake the small models into the image so startup needs no downloads.
# (Parler is ~3.5 GB, so it downloads on first use instead.)
RUN python -c "import stanza; stanza.download('hi', model_dir='/home/user/app/.cache/stanza', processors={'tokenize': 'hdtb', 'pos': 'hdtb_charlm'})" \
 && python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration as W; WhisperProcessor.from_pretrained('vasista22/whisper-hindi-small'); W.from_pretrained('vasista22/whisper-hindi-small')" \
 && python -c "from transformers import AutoTokenizer, VitsModel; AutoTokenizer.from_pretrained('facebook/mms-tts-hin'); VitsModel.from_pretrained('facebook/mms-tts-hin')"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
