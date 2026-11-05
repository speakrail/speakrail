# Speakrail model-init: downloads and patches the models into $MODELS_DIR, then exits (compose runs it before the GPU services).
FROM docker.io/library/python:3.12-slim
RUN pip install --no-cache-dir "huggingface_hub==1.31.0"
COPY init.py /app/init.py
ENV MODELS_DIR=/models PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/app/init.py"]
