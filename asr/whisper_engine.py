"""
Katman 1 — Whisper API ASR Motoru

OpenAI Whisper modeli küfürlerle gerçek veriyle eğitilmiştir.
Bu modül asıl transkripsiyon kaynağı olarak kullanılır.
word_timestamps=True ile kelime bazlı start/end süresi de döndürülür.

Çevrimdışı / ücretsiz alternatif için whisper_local_engine.py
(opsiyonel) oluşturulabilir; arayüz aynı kalır.
"""

import logging
from typing import Optional

import openai

from config.settings import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_RESPONSE_FORMAT,
)

logger = logging.getLogger(__name__)


def transcribe_with_timestamps(
    audio_file_path: str,
    language: Optional[str] = WHISPER_LANGUAGE,
) -> list[dict]:
    """
    OpenAI Whisper API kullanarak ses dosyasını kelime bazlı
    zaman damgalarıyla transkribe eder.

    Args:
        audio_file_path : Ses dosyasının yolu (mp3, wav, m4a, vb.)
        language        : BCP-47 dil kodu (örn. "tr"). None ise otomatik algılar.

    Returns:
        Her eleman şu anahtarları içeren bir sözlük listesi:
        - "word"  (str)   : Whisper'ın transkribe ettiği kelime
        - "start" (float) : Kelimenin başlangıç zamanı (saniye)
        - "end"   (float) : Kelimenin bitiş zamanı (saniye)

    Raises:
        openai.OpenAIError: API çağrısı başarısız olursa.
        FileNotFoundError : Ses dosyası bulunamazsa.
    """
    client = openai.OpenAI()

    logger.info("[WHISPER] '%s' transkribe ediliyor...", audio_file_path)

    with open(audio_file_path, "rb") as audio_file:
        kwargs: dict = {
            "model": WHISPER_MODEL,
            "file": audio_file,
            "response_format": WHISPER_RESPONSE_FORMAT,
            "timestamp_granularities": ["word"],
        }
        # Dil belirtilmişse ekle; None ise Whisper otomatik algılar
        if language:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    word_timestamps = _parse_whisper_response(response)
    logger.debug("[WHISPER] %d kelime transkribe edildi.", len(word_timestamps))
    return word_timestamps


def _parse_whisper_response(response) -> list[dict]:
    """
    Whisper API yanıtından kelime listesini standart formata çevirir.
    API'nin farklı response_format çıktılarını tolere eder.
    """
    word_timestamps: list[dict] = []

    # verbose_json → response.words listesi gelir
    if hasattr(response, "words") and response.words:
        for word_info in response.words:
            word_timestamps.append({
                "word":  word_info.word,
                "start": float(word_info.start),
                "end":   float(word_info.end),
            })
        return word_timestamps

    # Yedek: segments içindeki words listesi
    if hasattr(response, "segments") and response.segments:
        for segment in response.segments:
            if hasattr(segment, "words"):
                for word_info in segment.words:
                    word_timestamps.append({
                        "word":  word_info.get("word", ""),
                        "start": float(word_info.get("start", 0.0)),
                        "end":   float(word_info.get("end", 0.0)),
                    })

    return word_timestamps
