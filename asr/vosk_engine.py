"""
Katman 1 — Vosk ASR Motoru

Vosk burada yalnızca bir "zaman damgası makinesi" olarak çalışır.
Kelime doğruluğu Türkçe küfürler için düşük olduğundan transkripsiyon
amacıyla kullanılmaz; yalnızca zaman bilgisi (start/end) için koşulur.

Elde edilen zaman bilgisi daha sonra time_aligner tarafından
Whisper zaman eksenine hizalanır.
"""

import json
import logging
import os
import threading
import wave
from typing import Any

from vosk import KaldiRecognizer, Model, SetLogLevel

from config.settings import VOSK_CHUNK_SIZE, VOSK_MODEL_PATH

# Vosk'un C++ katmanından gelen gereksiz logları bastır
SetLogLevel(-1)

logger = logging.getLogger(__name__)

# Model bir kez yüklenir; modülün ömrü boyunca önbellekte tutulur
_vosk_model: Model | None = None
_vosk_model_lock = threading.Lock()


def _get_model() -> Model:
    """Singleton Vosk modelini döndürür. İlk çağrıda diskten yükler."""
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model
    with _vosk_model_lock:
        if _vosk_model is not None:
            return _vosk_model
        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(
                f"Vosk model dizini bulunamadı: '{VOSK_MODEL_PATH}'\n"
                "Lütfen Turkish Vosk modelini indirip 'model/' klasörüne çıkartın.\n"
                "İndirme: https://alphacephei.com/vosk/models"
            )
        logger.info("[VOSK] Model yükleniyor: %s", VOSK_MODEL_PATH)
        try:
            _vosk_model = Model(VOSK_MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Vosk modeli yüklenemedi: {exc}\n"
                "Model dizini bozuk veya eksik olabilir."
            ) from exc
    return _vosk_model


def get_word_timestamps(wav_file_path: str) -> list[dict[str, Any]]:
    """
    Bir WAV dosyasını Vosk ile analiz eder ve kelime bazlı
    zaman damgalarını döndürür.

    Args:
        wav_file_path: Mono, 16kHz WAV dosyasının yolu.

    Returns:
        Her eleman şu anahtarları içeren bir sözlük listesi:
        - "word"  (str)   : Vosk'un tanıdığı kelime
        - "start" (float) : Kelimenin başlangıç zamanı (saniye)
        - "end"   (float) : Kelimenin bitiş zamanı (saniye)
        - "conf"  (float) : Güven skoru (0.0–1.0)

    Raises:
        FileNotFoundError: WAV dosyası bulunamazsa.
        ValueError: Dosya Mono değilse.
        RuntimeError: Model yüklenemezse.
    """
    model = _get_model()

    with wave.open(wav_file_path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(
                f"Vosk yalnızca Mono (tek kanal) ses kabul eder. "
                f"'{wav_file_path}' dosyası {wf.getnchannels()} kanallı."
            )

        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetWords(True)  # Kelime bazlı zaman damgası aktif

        word_timestamps: list[dict[str, Any]] = []

        # Dosyayı parça parça işle
        while True:
            data = wf.readframes(VOSK_CHUNK_SIZE)
            if not data:
                break

            if recognizer.AcceptWaveform(data):
                try:
                    result = json.loads(recognizer.Result())
                except json.JSONDecodeError as exc:
                    logger.warning("[VOSK] JSON parse hatası (chunk atlandı): %s", exc)
                    continue
                _extract_words(result, word_timestamps)

        # Dosya sonunda kalan son segment
        try:
            final_result = json.loads(recognizer.FinalResult())
        except json.JSONDecodeError as exc:
            logger.warning("[VOSK] Final JSON parse hatası: %s", exc)
            final_result = {}
        _extract_words(final_result, word_timestamps)

    if not word_timestamps:
        logger.warning("[VOSK] Hiç kelime zaman damgası çıkarılamadı. Sessiz veya tanınamayan ses.")
    else:
        logger.debug("[VOSK] %d kelime zaman damgası çıkarıldı.", len(word_timestamps))
    return word_timestamps


def _extract_words(vosk_result: dict[str, Any], target_list: list[dict[str, Any]]) -> None:
    """
    Vosk JSON sonucundan kelime listesini çıkarıp target_list'e ekler.
    'result' anahtarı yoksa (sessiz bölüm) hiçbir şey eklenmez.
    """
    if "result" in vosk_result:
        for word_info in vosk_result["result"]:
            target_list.append({
                "word":  word_info.get("word", ""),
                "start": word_info.get("start", 0.0),
                "end":   word_info.get("end", 0.0),
                "conf":  word_info.get("conf", 1.0),
            })
