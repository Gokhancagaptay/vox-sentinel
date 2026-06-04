FROM python:3.10-slim

# FFmpeg — ses format dönüşümü için gerekli
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python bağımlılıklarını önce kopyala (layer cache optimizasyonu)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kaynak kodunu kopyala (model/ ve whisper_models/ hariç — volume ile mount edilir)
COPY . .

EXPOSE 8000

ENV VOXSENTINEL_WHISPER_MODE=local

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
