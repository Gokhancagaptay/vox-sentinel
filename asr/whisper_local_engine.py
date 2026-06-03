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
from typing import Optional

import whisper

from config.settings import (
    WHISPER_LOCAL_MODEL_SIZE,
    WHISPER_LOCAL_MODEL_DIR,
    WHISPER_LANGUAGE,
)

logger = logging.getLogger(__name__)

# Model bir kez yüklenir; modülün ömrü boyunca önbellekte tutulur
_local_model: whisper.Whisper | None = None


def _get_model() -> whisper.Whisper:
    """
    Singleton yerel Whisper modelini döndürür.
    İlk çağrıda WHISPER_LOCAL_MODEL_DIR'den yükler (veya indirir).
    """
    global _local_model
    if _local_model is None:
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
    language: Optional[str] = WHISPER_LANGUAGE,
) -> list[dict]:
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

    transcribe_kwargs: dict = {
        "word_timestamps": True,
        "verbose": False,          # Whisper'ın kendi progress çıktısını kapat
    }
    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(audio_file_path, **transcribe_kwargs)

    word_timestamps = _parse_local_whisper_result(result)
    logger.info(
        "[WHISPER LOCAL] %d kelime transkribe edildi.", len(word_timestamps)
    )
    return word_timestamps


def _parse_local_whisper_result(result: dict) -> list[dict]:
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
