"""
Katman 1 — Yerel Whisper ASR Motoru (Çevrimdışı / Ücretsiz)

openai-whisper paketi kullanılır (OpenAI API DEĞİL).
Model ilk çalıştırmada WHISPER_LOCAL_MODEL_DIR'e indirilir,
sonraki çalıştırmalarda diskten yüklenir.

CPU modunda çalışır; GPU varsa otomatik olarak kullanılır.

Performans rehberi (yaklaşık, CPU i5/i7 için):
    tiny   : 8 saniyelik ses ~2s   → hızlı ama Türkçe küfür doğruluğu düşük
    base   : 8 saniyelik ses ~6s   → iyi denge (varsayılan)
    small  : 8 saniyelik ses ~20s  → daha iyi doğruluk
    medium : 8 saniyelik ses ~60s  → yüksek doğruluk, sabırlı olun

Arayüz: whisper_engine.py ile birebir aynıdır.
        Pipeline her iki motoru da şeffaf şekilde kullanabilir.
"""

import logging
import os
import threading
from typing import Any

import numpy as np
import whisper

from config.settings import (
    WHISPER_LOCAL_MODEL_SIZE,
    WHISPER_LOCAL_MODEL_DIR,
    WHISPER_LANGUAGE,
    WHISPER_CHUNK_THRESHOLD_SEC,
    WHISPER_CHUNK_DURATION_SEC,
    WHISPER_CHUNK_OVERLAP_SEC,
)

logger = logging.getLogger(__name__)

# Model bir kez yüklenir; modülün ömrü boyunca önbellekte tutulur
_local_model: whisper.Whisper | None = None
_local_model_lock = threading.Lock()


def _get_model() -> whisper.Whisper:
    """
    Singleton yerel Whisper modelini döndürür.
    İlk çağrıda WHISPER_LOCAL_MODEL_DIR'den yükler (veya indirir).
    """
    global _local_model
    if _local_model is not None:
        return _local_model
    with _local_model_lock:
        if _local_model is not None:
            return _local_model
        try:
            os.makedirs(WHISPER_LOCAL_MODEL_DIR, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Whisper model dizini oluşturulamadı: '{WHISPER_LOCAL_MODEL_DIR}'\n"
                f"İzin hatası veya disk sorunu: {exc}"
            ) from exc

        logger.info(
            "[WHISPER LOCAL] '%s' modeli yükleniyor (dizin: %s)...",
            WHISPER_LOCAL_MODEL_SIZE,
            WHISPER_LOCAL_MODEL_DIR,
        )
        try:
            _local_model = whisper.load_model(
                WHISPER_LOCAL_MODEL_SIZE,
                download_root=WHISPER_LOCAL_MODEL_DIR,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Yerel Whisper modeli yüklenemedi (boyut: '{WHISPER_LOCAL_MODEL_SIZE}'): {exc}\n"
                "Olası nedenler: geçersiz model adı, disk alanı yetersiz, ağ hatası (ilk indirme)."
            ) from exc

        logger.info("[WHISPER LOCAL] Model hazır.")
    return _local_model


def transcribe_with_timestamps(
    audio_file_path: str,
    language: str | None = WHISPER_LANGUAGE,
) -> list[dict[str, Any]]:
    """
    Yerel Whisper modeli kullanarak ses dosyasını kelime bazlı
    zaman damgalarıyla transkribe eder.

    whisper_engine.py ile aynı imzaya sahiptir; pipeline her ikisini
    şeffaf şekilde kullanabilir.

    Args:
        audio_file_path : Ses dosyasının yolu (wav, mp3, vb.)
        language        : BCP-47 dil kodu (örn. "tr"). None ise otomatik algılar.

    Returns:
        Her eleman şu anahtarları içeren bir sözlük listesi:
        - "word"  (str)   : Transkripsiyon kelimesi
        - "start" (float) : Başlangıç zamanı (saniye)
        - "end"   (float) : Bitiş zamanı (saniye)

    Raises:
        FileNotFoundError: Ses dosyası bulunamazsa.
        RuntimeError     : Model yüklenemezse.
    """
    model = _get_model()

    logger.info(
        "[WHISPER LOCAL] '%s' transkribe ediliyor (model: %s)...",
        audio_file_path,
        WHISPER_LOCAL_MODEL_SIZE,
    )

    # Ses dizisini yükle; whisper.load_audio() → 16kHz mono float32
    audio_array: np.ndarray = whisper.load_audio(audio_file_path)
    sample_rate = whisper.audio.SAMPLE_RATE  # 16000
    duration_sec = len(audio_array) / sample_rate

    if duration_sec > WHISPER_CHUNK_THRESHOLD_SEC:
        logger.info(
            "[WHISPER LOCAL] Uzun dosya (%.1fs > %ds); parçalı işleme başlatılıyor.",
            duration_sec, WHISPER_CHUNK_THRESHOLD_SEC,
        )
        word_timestamps = _transcribe_chunked(model, audio_array, sample_rate, language)
    else:
        transcribe_kwargs: dict[str, Any] = {
            "word_timestamps": True,
            "verbose": False,
        }
        if language:
            transcribe_kwargs["language"] = language
        result = model.transcribe(audio_file_path, **transcribe_kwargs)
        word_timestamps = _parse_local_whisper_result(result)

    logger.info(
        "[WHISPER LOCAL] %d kelime transkribe edildi.", len(word_timestamps)
    )
    return word_timestamps


def _transcribe_chunked(
    model: whisper.Whisper,
    audio_array: np.ndarray,
    sample_rate: int,
    language: str | None,
) -> list[dict[str, Any]]:
    """Uzun sesi örtüşen parçalara bölerek transkribe eder."""
    chunk_samples   = int(WHISPER_CHUNK_DURATION_SEC * sample_rate)
    overlap_samples = int(WHISPER_CHUNK_OVERLAP_SEC  * sample_rate)
    stride          = chunk_samples - overlap_samples

    all_words: list[dict[str, Any]] = []
    offset = 0

    while offset < len(audio_array):
        chunk = audio_array[offset : offset + chunk_samples]
        chunk_start_sec = offset / sample_rate

        transcribe_kwargs: dict[str, Any] = {"word_timestamps": True, "verbose": False}
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(chunk, **transcribe_kwargs)

        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                clean_word = word_info.get("word", "").strip()
                if clean_word:
                    all_words.append({
                        "word":  clean_word,
                        "start": float(word_info.get("start", 0.0)) + chunk_start_sec,
                        "end":   float(word_info.get("end",   0.0)) + chunk_start_sec,
                    })

        offset += stride

    return _dedup_words_by_time(all_words)


def _dedup_words_by_time(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Örtüşme bölgesindeki tekrar eden kelimeleri 50ms toleransla eler."""
    if not words:
        return []
    words.sort(key=lambda w: w["start"])
    deduped = [words[0]]
    for w in words[1:]:
        if w["start"] - deduped[-1]["start"] > 0.05:
            deduped.append(w)
    return deduped


def _parse_local_whisper_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Yerel Whisper çıktısını standart {word, start, end} formatına çevirir.

    Yerel Whisper kelime bilgisini result["segments"][i]["words"] içinde tutar.
    Her kelimede {"word", "start", "end", "probability"} anahtarları gelir.
    """
    word_timestamps: list[dict] = []

    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            # Whisper bazen kelimenin başına boşluk ekler; temizle
            clean_word = word_info.get("word", "").strip()
            if clean_word:
                word_timestamps.append({
                    "word":  clean_word,
                    "start": float(word_info.get("start", 0.0)),
                    "end":   float(word_info.get("end",   0.0)),
                })

    return word_timestamps
