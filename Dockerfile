FROM python:3.11-slim

WORKDIR /app

# Dépendances système nécessaires pour torchaudio / librosa / coqui-tts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir "torch<2.9" "torchaudio<2.9" && \
    pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=200", "--server.enableXsrfProtection=false"]