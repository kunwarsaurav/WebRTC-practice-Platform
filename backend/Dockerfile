FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg default-jre && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm || true

COPY . .

# Render expects the web service to bind to the $PORT environment variable (defaults to 10000)
EXPOSE 10000

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"
