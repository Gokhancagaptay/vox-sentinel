"""
Katman 3 — Ses Sansür İşlemcisi

Karar motoru hangi segmentlerin sansürleneceğini belirlediğinde,
bu modül pydub/FFmpeg kullanarak ilgili zaman aralıklarını bip sesiyle değiştirir.

Diyagramdan: start_ms → end_ms aralığı sine wave ile değiştirilir.

Performans: Segmentler soldan sağa tek geçişte birleştirilerek O(n) ses inşa edilir.
Önceki ters-sıra yaklaşımı her adımda tam AudioSegment kopyası oluşturuyordu (O(n²)).
"""

import logging
from typing import Any

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.generators import Sine

from config.settings import (
    BEEP_FREQUENCY_HZ,
    BEEP_GAIN_DB,
    CENSOR_PADDING_MS,
    MIN_SEGMENT_DURATION_MS,
)

logger = logging.getLogger(__name__)


def _merge_overlapping(segments: list[dict[str, Any]], audio_duration_ms: int) -> list[dict[str, Any]]:
    """
    Padding uygulandıktan sonra çakışan veya bitişik segmentleri birleştirir.
    Giriş listesi start_ms'e göre sıralı olmalıdır.
    """
    merged: list[dict] = []
    for seg in segments:
        start_ms = max(0, seg["start_ms"] - CENSOR_PADDING_MS)
        end_ms   = min(audio_duration_ms, seg["end_ms"] + CENSOR_PADDING_MS)

        if end_ms - start_ms < MIN_SEGMENT_DURATION_MS:
            logger.warning(
                "[SES] Minimum sürenin altındaki segment atlandı: start=%d end=%d",
                start_ms, end_ms,
            )
            continue

        if merged and start_ms <= merged[-1]["end_ms"]:
            # Mevcut segmentle çakışıyor; genişlet
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], end_ms)
            merged[-1]["word"] = f"{merged[-1].get('word', '')}+{seg.get('word', '')}"
        else:
            merged.append({**seg, "start_ms": start_ms, "end_ms": end_ms})

    return merged


def apply_censor_beeps(
    audio_file_path: str,
    censor_segments: list[dict[str, Any]],
    output_path: str,
) -> str:
    """
    Ses dosyasındaki belirtilen zaman aralıklarını 1kHz bip sesiyle değiştirir.

    Tek geçişli O(n) algoritma: çıkış, sessiz bölge + beep + sessiz bölge + ...
    parçalarından soldan sağa birleştirilerek inşa edilir.

    Args:
        audio_file_path  : Giriş ses dosyasının yolu (wav, mp3, vb.)
        censor_segments  : voting_engine'den gelen final sansür listesi.
                           Her eleman: {"start_ms", "end_ms", ...}
        output_path      : Sansürlü çıkış dosyasının kaydedileceği yol.

    Returns:
        Kaydedilen dosyanın yolu (output_path ile aynı).

    Raises:
        FileNotFoundError: Giriş dosyası bulunamazsa.
        ValueError       : censor_segments boşsa.
    """
    if not censor_segments:
        raise ValueError("Sansürlenecek segment listesi boş.")

    logger.info("[SES] '%s' yükleniyor...", audio_file_path)
    try:
        audio = AudioSegment.from_file(audio_file_path)
    except (CouldntDecodeError, FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"Ses dosyası yüklenemedi: '{audio_file_path}'\nHata: {exc}"
        ) from exc

    audio_duration_ms = len(audio)

    # Başa göre sırala → çakışanları merge et → tek geçişte uygula
    sorted_segments = sorted(censor_segments, key=lambda s: s["start_ms"])
    active_segments = _merge_overlapping(sorted_segments, audio_duration_ms)

    if not active_segments:
        logger.warning("[SES] Tüm segmentler geçersiz/çakışma sonrası boş kaldı.")
        output_format = output_path.rsplit(".", 1)[-1].lower()
        audio.export(output_path, format=output_format)
        return output_path

    # ─── Tek geçişli birleştirme ───────────────────────────────────
    # result = ses[0:seg1.start] + beep1 + ses[seg1.end:seg2.start] + beep2 + ...
    result = AudioSegment.empty()
    prev_end = 0

    for segment in active_segments:
        start_ms = segment["start_ms"]
        end_ms   = segment["end_ms"]
        duration_ms = end_ms - start_ms

        # Sansür öncesi temiz ses bölümü
        result += audio[prev_end:start_ms]

        # Bip sesi
        beep = (
            Sine(BEEP_FREQUENCY_HZ)
            .to_audio_segment(duration=duration_ms)
            .apply_gain(BEEP_GAIN_DB)
        )
        result += beep
        prev_end = end_ms

        logger.debug(
            "[SES] Sansür uygulandı: %dms → %dms ('%s')",
            start_ms, end_ms, segment.get("word", "?"),
        )

    # Son temiz bölüm
    result += audio[prev_end:]

    # Giriş formatını koruyarak kaydet
    output_format = output_path.rsplit(".", 1)[-1].lower()
    result.export(output_path, format=output_format)
    logger.info("[SES] Sansürlü dosya kaydedildi: '%s'", output_path)

    return output_path
