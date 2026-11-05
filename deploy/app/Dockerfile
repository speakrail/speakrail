# Speakrail app: the harness and the web UI (no GPU).
FROM docker.io/library/python:3.12-slim
WORKDIR /app
COPY deploy/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY deploy/app/start.sh ./start.sh
RUN chmod +x start.sh
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src UI_HOST=0.0.0.0 UI_PORT=8080 VOICE=/app/src/speakrail/harness/voices/female_a.wav \
    SPEAKRAIL_TOKENIZER_DIR=/models/speakrail-gemma/tokenizer PHRASE_CACHE_DIR=/data/phrase_cache
EXPOSE 8080
ENTRYPOINT ["/app/start.sh"]
