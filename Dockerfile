FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRADING_PF_CONFIG=config/default.yaml

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

# Use hosting platform assigned PORT (Render/Railway/Fly/etc.)
CMD sh -c "gunicorn wsgi:app --workers 1 --threads 2 --timeout 240 --bind 0.0.0.0:${PORT:-5000}"
