"""
Katman 1 — Whisper API ASR Motoru

OpenAI Whisper modeli küfürlerle gerçek veriyle eğitilmiştir.
Bu modül asıl transkripsiyon kaynağı olarak kullanılır.
word_timestamps=True ile kelime bazlı start/end süresi de döndürülür.

Çevrimdışı / ücretsiz alternatif için whisper_local_engine.py
(opsiyonel) oluşturulabilir; arayüz aynı kalır.
"""

import logging
import os
import time
from typing import Any

import openai

from config.settings import (
    WHISPER_API_TIMEOUT,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    WHISPER_RESPONSE_FORMAT,
    WHISPER_RETRY_MAX,
)

logger = logging.getLogger(__name__)


def transcribe_with_timestamps(
    audio_file_path: str,
    language: str | None = WHISPER_LANGUAGE,
) -> list[dict[str, Any]]:
    """
    OpenAI Whisper API kullanarak ses dosyasını kelime bazlı
    zaman damgalarıyla transkribe eder.

    API hatalarında exponential backoff ile WHISPER_RETRY_MAX kez yeniden dener.

    Args:
        audio_file_path : Ses dosyasının yolu (mp3, wav, m4a, vb.)
        language        : BCP-47 dil kodu (örn. "tr"). None ise otomatik algılar.

    Returns:
        Her eleman şu anahtarları içeren bir sözlük listesi:
        - "word"  (str)   : Whisper'ın transkribe ettiği kelime
        - "start" (float) : Kelimenin başlangıç zamanı (saniye)
        - "end"   (float) : Kelimenin bitiş zamanı (saniye)

    Raises:
        RuntimeError      : OPENAI_API_KEY eksikse.
        openai.OpenAIError: Tüm denemeler başarısız olursa.
        FileNotFoundError : Ses dosyası bulunamazsa.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY ortam değişkeni bulunamadı.\n"
            "Çevrimdışı kullanım için settings.py'de WHISPER_MODE='local' yapın."
        )

    client = openai.OpenAI(timeout=WHISPER_API_TIMEOUT)

    logger.info("[WHISPER] '%s' transkribe ediliyor...", audio_file_path)

    last_exc: Exception | None = None
    for attempt in range(1, WHISPER_RETRY_MAX + 1):
        try:
            with open(audio_file_path, "rb") as audio_file:
                kwargs: dict = {
                    "model": WHISPER_MODEL,
                    "file": audio_file,
                    "response_format": WHISPER_RESPONSE_FORMAT,
                    "timestamp_granularities": ["word"],
                }
                if language:
                    kwargs["language"] = language

                response = client.audio.transcriptions.create(**kwargs)

            word_timestamps = _parse_whisper_response(response)
            logger.debug("[WHISPER] %d kelime transkribe edildi.", len(word_timestamps))
            return word_timestamps

        except openai.OpenAIError as exc:
            last_exc = exc
            if attempt < WHISPER_RETRY_MAX:
                wait = 2 ** attempt
                logger.warning(
                    "[WHISPER] API hatası (deneme %d/%d): %s. %ds sonra tekrar...",
                    attempt, WHISPER_RETRY_MAX, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "[WHISPER] Tüm %d deneme başarısız: %s", WHISPER_RETRY_MAX, exc
                )

    raise last_exc  # type: ignore[misc]


def _parse_whisper_response(response: Any) -> list[dict[str, Any]]:
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
