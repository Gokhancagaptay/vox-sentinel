"""
Katman 1 — faster-whisper ASR Motoru (Üretim Backend'i)

faster-whisper, openai-whisper'ın CTranslate2 tabanlı yeniden uygulamasıdır.
Aynı Whisper modellerini kullanır; CPU'da ~4× daha hızlı, ~2× daha az bellek.

Neden faster-whisper?
- INT8 quantization → CPU inferans süresi dramatik düşer
- Kelime bazlı timestamp desteği (word_timestamps=True)
- asyncio.to_thread() ile FastAPI async pipeline'ına tam uyumlu
- openai-whisper ile birebir aynı çıktı formatı → pipeline değişmez

Modeller HuggingFace'den CTranslate2 formatında otomatik indirilir.
Dizin: WHISPER_LOCAL_MODEL_DIR (varsayılan: whisper_models/)

Arayüz: whisper_local_engine.py ile aynıdır.
"""

import logging
import threading
from typing import Any

from config.settings import (
    WHISPER_CHUNK_DURATION_SEC,
    WHISPER_CHUNK_OVERLAP_SEC,
    WHISPER_CHUNK_THRESHOLD_SEC,
    WHISPER_LANGUAGE,
    WHISPER_LOCAL_MODEL_DIR,
    WHISPER_LOCAL_MODEL_SIZE,
)

logger = logging.getLogger(__name__)

_fw_model = None
_fw_model_lock = threading.Lock()

# faster-whisper model dizini — openai-whisper'dan ayrı alt dizin
_FW_MODEL_DIR = WHISPER_LOCAL_MODEL_DIR + "/faster_whisper"


def _get_model():
    """Singleton faster-whisper modelini döndürür. Double-checked locking."""
    global _fw_model
    if _fw_model is not None:
        return _fw_model
    with _fw_model_lock:
        if _fw_model is not None:
            return _fw_model

        from faster_whisper import WhisperModel

        logger.info(
            "[FASTER-WHISPER] '%s' modeli yükleniyor (dizin: %s)...",
            WHISPER_LOCAL_MODEL_SIZE,
            _FW_MODEL_DIR,
        )
        _fw_model = WhisperModel(
            WHISPER_LOCAL_MODEL_SIZE,
            device="cpu",
            compute_type="int8",           # CPU için INT8 quantization
            download_root=_FW_MODEL_DIR,
        )
        logger.info("[FASTER-WHISPER] Model hazır.")
    return _fw_model


def transcribe_with_timestamps(
    audio_file_path: str,
    language: str | None = WHISPER_LANGUAGE,
) -> list[dict[str, Any]]:
    """
    faster-whisper ile ses dosyasını kelime bazlı zaman damgalarıyla transkribe eder.

    Args:
        audio_file_path : Ses dosyasının yolu (wav, mp3, vb.)
        language        : BCP-47 dil kodu ("tr"). None ise otomatik algılar.

    Returns:
        [{"word": str, "start": float, "end": float}, ...]
    """
    import os
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: '{audio_file_path}'")

    model = _get_model()

    logger.info(
        "[FASTER-WHISPER] '%s' transkribe ediliyor (model: %s)...",
        audio_file_path,
        WHISPER_LOCAL_MODEL_SIZE,
    )

    # Süre kontrolü — uzun dosyalar parçalanır
    duration_sec = _get_audio_duration(audio_file_path)

    if duration_sec > WHISPER_CHUNK_THRESHOLD_SEC:
        logger.info(
            "[FASTER-WHISPER] Uzun dosya (%.1fs); parçalı işleme...", duration_sec
        )
        words = _transcribe_chunked(model, audio_file_path, language)
    else:
        words = _transcribe_single(model, audio_file_path, language)

    logger.info("[FASTER-WHISPER] %d kelime transkribe edildi.", len(words))
    return words


def _transcribe_single(
    model,
    audio_file_path: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """Tek parça transkripsiyon."""
    segments, _ = model.transcribe(
        audio_file_path,
        language=language or "tr",
        word_timestamps=True,
        vad_filter=True,           # Sessiz bölgeleri atla (hız kazanımı)
        vad_parameters={"min_silence_duration_ms": 500},
    )

    words: list[dict[str, Any]] = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                clean = w.word.strip()
                if clean:
                    words.append({
                        "word":  clean,
                        "start": float(w.start),
                        "end":   float(w.end),
                    })
    return words


def _transcribe_chunked(
    model,
    audio_file_path: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """Uzun dosyaları örtüşen parçalarda transkribe eder."""
    import os
    import tempfile

    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_file_path)
    chunk_ms   = int(WHISPER_CHUNK_DURATION_SEC * 1000)
    overlap_ms = int(WHISPER_CHUNK_OVERLAP_SEC  * 1000)
    stride_ms  = chunk_ms - overlap_ms

    all_words: list[dict[str, Any]] = []
    offset_ms = 0

    while offset_ms < len(audio):
        chunk = audio[offset_ms : offset_ms + chunk_ms]
        chunk_start_sec = offset_ms / 1000.0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                         prefix="fw_chunk_") as tmp:
            chunk.export(tmp.name, format="wav")
            tmp_path = tmp.name

        try:
            chunk_words = _transcribe_single(model, tmp_path, language)
        finally:
            os.unlink(tmp_path)

        for w in chunk_words:
            all_words.append({
                "word":  w["word"],
                "start": w["start"] + chunk_start_sec,
                "end":   w["end"]   + chunk_start_sec,
            })

        offset_ms += stride_ms

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


def _get_audio_duration(audio_file_path: str) -> float:
    """Ses dosyasının süresini saniye cinsinden döndürür."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / 1000.0
