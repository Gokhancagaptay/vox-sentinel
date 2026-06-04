"""
Ana Orkestratör — Üç Katmanlı Sansür Boru Hattı

Bu modül diyagramdaki tüm akışı tek bir fonksiyonda birleştirir:

    Katman 1 (Paralel ASR):
        ┌─ Vosk          → Kelime zaman damgaları
        ├─ Whisper API   → Doğru transkripsiyon + zaman damgaları
        └─ Fonetik       → Vosk OOV tespiti (fuzzy match)

    Katman 2 (Karar):
        ├─ Zaman hizalama   → Vosk ts'yi Whisper ts eksenine hizala
        └─ Çok-kaynaklı oy  → Whisper ∨ Fonetik → sansür

    Katman 3 (Ses):
        └─ FFmpeg/pydub → start_ms–end_ms'i sine wave ile değiştir

Her katman bağımsız modüllerde yaşar; bu dosya yalnızca aralarındaki
veri akışını düzenler.
"""

import asyncio
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from asr.vosk_engine      import get_word_timestamps
from asr.phonetic_matcher import scan_for_phonetic_matches
from decision.time_aligner  import build_anchor_map, align_phonetic_detections
from decision.voting_engine import find_whisper_banned_words, vote_and_merge
from audio.censor_processor import apply_censor_beeps
from audio.converter        import to_vosk_wav, cleanup_temp_wav
from config.settings import WHISPER_MODE, WHISPER_LOCAL_BACKEND

logger = logging.getLogger(__name__)


def _resolve_whisper_transcriber() -> tuple[Callable[..., list[dict[str, Any]]], str]:
    """
    WHISPER_MODE ayarına ve OPENAI_API_KEY varlığına göre
    doğru transcribe_with_timestamps fonksiyonunu seçer.

    Dönen değer: (transcribe_fn, kaynak_adı)
    """
    mode = WHISPER_MODE.lower()

    if mode == "api":
        from asr.whisper_engine import transcribe_with_timestamps
        return transcribe_with_timestamps, "api"

    if mode == "local":
        return _resolve_local_transcriber()

    # mode == "auto": API anahtarı varsa API, yoksa yerel
    if os.environ.get("OPENAI_API_KEY"):
        from asr.whisper_engine import transcribe_with_timestamps
        return transcribe_with_timestamps, "api"

    return _resolve_local_transcriber()


def _resolve_local_transcriber() -> tuple[Callable[..., list[dict[str, Any]]], str]:
    """
    WHISPER_LOCAL_BACKEND ayarına göre yerel transkripsiyon fonksiyonunu seçer.
    faster-whisper kurulu değilse openai-whisper'a düşer.
    """
    if WHISPER_LOCAL_BACKEND == "faster-whisper":
        try:
            from asr.faster_whisper_engine import transcribe_with_timestamps
            return transcribe_with_timestamps, "faster-whisper"
        except ImportError:
            logger.warning(
                "[PIPELINE] faster-whisper kurulu değil; openai-whisper'a düşülüyor. "
                "Kurmak için: pip install faster-whisper"
            )

    from asr.whisper_local_engine import transcribe_with_timestamps
    return transcribe_with_timestamps, "local"


@dataclass
class PipelineResult:
    """
    Boru hattı çalışmasının tüm ara ve son sonuçlarını taşır.
    Hata ayıklama ve raporlama için her katmanın çıktısı saklanır.
    """
    vosk_words:              list[dict] = field(default_factory=list)
    whisper_words:           list[dict] = field(default_factory=list)
    phonetic_detections:     list[dict] = field(default_factory=list)
    whisper_detections:      list[dict] = field(default_factory=list)
    anchor_count:            int = 0
    final_censor_segments:   list[dict] = field(default_factory=list)
    output_file:             str | None = None
    censored:                bool = False


def run_censorship_pipeline(
    audio_file_path: str,
    output_path: str,
    use_vosk: bool = True,
    use_whisper: bool = True,
    use_phonetic: bool = True,
) -> PipelineResult:
    """
    Üç katmanlı sansür boru hattını çalıştırır.

    Args:
        audio_file_path : İşlenecek ses dosyasının yolu.
                          WAV (Vosk için gerekli), MP3, M4A vb. desteklenir.
        output_path     : Sansürlü çıkış dosyasının yolu.
        use_vosk        : Vosk zaman damgası motoru aktif mi?
        use_whisper     : Whisper API transkripsiyon motoru aktif mi?
        use_phonetic    : Fonetik güvenlik ağı aktif mi?
                          (Yalnızca use_vosk=True ile çalışır)

    Returns:
        PipelineResult: Tüm ara sonuçları ve final çıktıyı içerir.

    Not:
        Whisper API için OPENAI_API_KEY ortam değişkeni set edilmiş olmalıdır.
        Vosk için proje kökünde 'model/' klasörü bulunmalıdır.
    """
    result = PipelineResult()

    # ═══════════════════════════════════════════════════════════════
    # ÖN İŞLEM — Vosk için format dönüştürme (MP3, M4A vb. → WAV)
    # ═══════════════════════════════════════════════════════════════
    vosk_wav_path, is_temp_wav = audio_file_path, False
    if use_vosk:
        vosk_wav_path, is_temp_wav = to_vosk_wav(audio_file_path)

    try:
        _run_pipeline_layers(
            result, audio_file_path, vosk_wav_path,
            output_path, use_vosk, use_whisper, use_phonetic,
        )
    finally:
        # Geçici WAV dosyasını her durumda temizle (hata olsa bile)
        cleanup_temp_wav(vosk_wav_path, is_temp_wav)

    return result


async def run_censorship_pipeline_async(
    audio_file_path: str,
    output_path: str,
    use_vosk: bool = True,
    use_whisper: bool = True,
    use_phonetic: bool = True,
) -> PipelineResult:
    """
    run_censorship_pipeline'ın async versiyonu.
    Vosk ve Whisper çağrıları asyncio.to_thread() ile paralel çalışır.
    FastAPI gibi async framework'lerde kullanım için.
    """
    result = PipelineResult()

    vosk_wav_path, is_temp_wav = audio_file_path, False
    if use_vosk:
        vosk_wav_path, is_temp_wav = await asyncio.to_thread(to_vosk_wav, audio_file_path)

    try:
        await _run_pipeline_layers_async(
            result, audio_file_path, vosk_wav_path,
            output_path, use_vosk, use_whisper, use_phonetic,
        )
    finally:
        await asyncio.to_thread(cleanup_temp_wav, vosk_wav_path, is_temp_wav)

    return result


async def _run_pipeline_layers_async(
    result: PipelineResult,
    audio_file_path: str,
    vosk_wav_path: str,
    output_path: str,
    use_vosk: bool,
    use_whisper: bool,
    use_phonetic: bool,
) -> None:
    """Katman 1'i (Vosk + Whisper) asyncio.gather ile paralel çalıştırır."""

    # ── Katman 1: Vosk ve Whisper paralel ──────────────────────────
    tasks: list[Any] = []
    task_names: list[str] = []

    if use_vosk:
        tasks.append(asyncio.to_thread(get_word_timestamps, vosk_wav_path))
        task_names.append("vosk")

    if use_whisper:
        transcribe_fn, whisper_source = _resolve_whisper_transcriber()
        tasks.append(asyncio.to_thread(transcribe_fn, audio_file_path))
        task_names.append("whisper")
        logger.info(
            "── KATMAN 1 | WHISPER [%s] (async) ─────────────────────", whisper_source.upper()
        )

    layer1_results = await asyncio.gather(*tasks, return_exceptions=True)

    for name, res in zip(task_names, layer1_results):
        if isinstance(res, Exception):
            logger.error("[%s] Hata: %s", name.upper(), res)
        elif name == "vosk":
            result.vosk_words = res
            logger.info("[VOSK] %d kelime zaman damgası çıkarıldı.", len(res))
        elif name == "whisper":
            result.whisper_words = res
            logger.info("[WHISPER] %d kelime transkribe edildi.", len(res))

    # ── Fonetik + Karar + Ses katmanları (sync, hızlı pure-Python) ──
    if use_vosk and use_phonetic and result.vosk_words:
        result.phonetic_detections = scan_for_phonetic_matches(result.vosk_words)

    _run_decision_and_audio_layers(result, audio_file_path, output_path, use_whisper, use_vosk)


def _run_decision_and_audio_layers(
    result: PipelineResult,
    audio_file_path: str,
    output_path: str,
    use_whisper: bool,
    use_vosk: bool,
) -> None:
    """Katman 2 (karar) ve Katman 3 (ses) — hem sync hem async pipeline paylaşır."""
    logger.info("── KATMAN 2 | KARAR MOTORU ──────────────────────────")

    if use_whisper and result.whisper_words:
        result.whisper_detections = find_whisper_banned_words(result.whisper_words)
        logger.info(
            "[KARAR] Whisper %d yasaklı kelime tespit etti.", len(result.whisper_detections)
        )

    aligned_phonetic = result.phonetic_detections
    if use_vosk and use_whisper and result.vosk_words and result.whisper_words:
        anchors = build_anchor_map(result.vosk_words, result.whisper_words)
        result.anchor_count = len(anchors)
        aligned_phonetic = align_phonetic_detections(result.phonetic_detections, anchors)

    result.final_censor_segments = vote_and_merge(result.whisper_detections, aligned_phonetic)
    logger.info("[OYLAMA] %d segment sansürlenecek.", len(result.final_censor_segments))

    logger.info("── KATMAN 3 | SES İŞLEMİ ───────────────────────────")
    if result.final_censor_segments:
        result.output_file = apply_censor_beeps(
            audio_file_path, result.final_censor_segments, output_path
        )
        result.censored = True
        logger.info("[SES] Sansürlü dosya: '%s'", result.output_file)
    else:
        logger.info("[SES] Sansürlenecek kelime bulunamadı. Dosya değiştirilmedi.")


def _run_pipeline_layers(
    result: PipelineResult,
    audio_file_path: str,
    vosk_wav_path: str,
    output_path: str,
    use_vosk: bool,
    use_whisper: bool,
    use_phonetic: bool,
) -> None:
    """Pipeline'ın üç katmanını sırayla çalıştırır (iç yardımcı fonksiyon)."""

    # ═══════════════════════════════════════════════════════════════
    # KATMAN 1 — Paralel ASR Motorları
    # ═══════════════════════════════════════════════════════════════

    # ── Vosk: Kelime bazlı zaman damgaları ─────────────────────────
    if use_vosk:
        logger.info("── KATMAN 1 | VOSK ──────────────────────────────────")
        result.vosk_words = get_word_timestamps(vosk_wav_path)
        logger.info("[VOSK] %d kelime zaman damgası çıkarıldı.", len(result.vosk_words))

    # ── Whisper (API veya Yerel): Doğru transkripsiyon + zaman damgaları ──
    if use_whisper:
        transcribe_fn, whisper_source = _resolve_whisper_transcriber()
        logger.info(
            "── KATMAN 1 | WHISPER [%s] ──────────────────────────", whisper_source.upper()
        )
        result.whisper_words = transcribe_fn(audio_file_path)
        logger.info(
            "[WHISPER %s] %d kelime transkribe edildi.",
            whisper_source.upper(), len(result.whisper_words),
        )

    # ── Fonetik eşleştirici: Vosk OOV tespiti ──────────────────────
    if use_vosk and use_phonetic and result.vosk_words:
        logger.info("── KATMAN 1 | FONETİK ───────────────────────────────")
        result.phonetic_detections = scan_for_phonetic_matches(result.vosk_words)
        logger.info(
            "[FONETİK] %d fonetik eşleşme bulundu.",
            len(result.phonetic_detections),
        )

    # ═══════════════════════════════════════════════════════════════
    # KATMAN 2 — Karar Motoru
    # ═══════════════════════════════════════════════════════════════

    logger.info("── KATMAN 2 | KARAR MOTORU ──────────────────────────")

    # ── Whisper doğrudan tespitleri ─────────────────────────────────
    if use_whisper and result.whisper_words:
        result.whisper_detections = find_whisper_banned_words(result.whisper_words)
        logger.info(
            "[KARAR] Whisper %d yasaklı kelime tespit etti.",
            len(result.whisper_detections),
        )

    # ── Fonetik tespitleri Whisper zaman eksenine hizala ────────────
    aligned_phonetic = result.phonetic_detections
    if use_vosk and use_whisper and result.vosk_words and result.whisper_words:
        anchors = build_anchor_map(result.vosk_words, result.whisper_words)
        result.anchor_count = len(anchors)
        logger.info("[HİZALAMA] %d anchor nokta bulundu.", result.anchor_count)

        aligned_phonetic = align_phonetic_detections(
            result.phonetic_detections, anchors
        )

    # ── Çok-kaynaklı oylama → final sansür listesi ─────────────────
    result.final_censor_segments = vote_and_merge(
        result.whisper_detections,
        aligned_phonetic,
    )
    logger.info(
        "[OYLAMA] %d segment sansürlenecek.", len(result.final_censor_segments)
    )

    # ═══════════════════════════════════════════════════════════════
    # KATMAN 3 — Ses Sansür İşlemi
    # ═══════════════════════════════════════════════════════════════

    logger.info("── KATMAN 3 | SES İŞLEMİ ───────────────────────────")

    if result.final_censor_segments:
        result.output_file = apply_censor_beeps(
            audio_file_path,
            result.final_censor_segments,
            output_path,
        )
        result.censored = True
        logger.info("[SES] Sansürlü dosya: '%s'", result.output_file)
    else:
        logger.info("[SES] Sansürlenecek kelime bulunamadı. Dosya değiştirilmedi.")
