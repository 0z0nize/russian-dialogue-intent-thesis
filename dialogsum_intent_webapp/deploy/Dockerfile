# Dockerfile для демо DialogSum-RU intent webapp.
# Контекст сборки — корень проекта `dialogsum_intent_webapp/`.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Системные зависимости: ffmpeg для аудио, libsndfile/libgomp на всякий случай.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Сначала requirements, чтобы лучше кешировалось
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Копируем проект
COPY app.py /app/app.py
COPY model_pipeline.py /app/model_pipeline.py
COPY models /app/models
COPY assets /app/assets
COPY README.md /app/README.md

EXPOSE 7860

CMD ["python", "app.py"]
