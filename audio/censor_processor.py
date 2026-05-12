"""
Katman 3 — Ses Sansür İşlemcisi

Karar motoru hangi segmentlerin sansürleneceğini belirlediğinde,
bu modül pydub/FFmpeg kullanarak ilgili zaman aralıklarını bip sesiyle değiştirir.

Diyagramdan: start_ms → end_ms aralığı sine wave ile değiştirilir.

Önemli: Segmentler sondan başa sıralı işlenir.
Bu, erken kesilen bir segmentin sonraki segmentlerin
milisaniye offsetini bozmasını önler.
"""

import logging
from pydub import AudioSegment
from pydub.generators import Sine

from config.settings import BEEP_FREQUENCY_HZ, BEEP_GAIN_DB, CENSOR_PADDING_MS

logger = logging.getLogger(__name__)


def apply_censor_beeps(
    audio_file_path: str,
    censor_segments: list[dict],
    output_path: str,
) -> str:
    """
    Ses dosyasındaki belirtilen zaman aralıklarını 1kHz bip sesiyle değiştirir.

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
    audio = AudioSegment.from_file(audio_file_path)
    audio_duration_ms = len(audio)

    # Zaman kaymasını önlemek için sondan başa sırala
    sorted_segments = sorted(
        censor_segments, key=lambda seg: seg["start_ms"], reverse=True
    )

    for segment in sorted_segments:
        # Padding uygula ve sınırların dışına çıkmasını engelle
        start_ms = max(0, segment["start_ms"] - CENSOR_PADDING_MS)
        end_ms   = min(audio_duration_ms, segment["end_ms"] + CENSOR_PADDING_MS)
        duration_ms = end_ms - start_ms

        if duration_ms <= 0:
            logger.warning(
                "[SES] Geçersiz segment atlandı: start=%d end=%d",
                start_ms, end_ms,
            )
            continue

        # Klasik TV sansür bip sesi: 1000 Hz sine wave
        beep = (
            Sine(BEEP_FREQUENCY_HZ)
            .to_audio_segment(duration=duration_ms)
            .apply_gain(BEEP_GAIN_DB)
        )

        audio = audio[:start_ms] + beep + audio[end_ms:]

        logger.debug(
            "[SES] Sansür uygulandı: %dms → %dms ('%s')",
            start_ms, end_ms, segment.get("word", "?"),
        )

    # Giriş formatını koruyarak kaydet
    output_format = output_path.rsplit(".", 1)[-1].lower()
    audio.export(output_path, format=output_format)
    logger.info("[SES] Sansürlü dosya kaydedildi: '%s'", output_path)

    return output_path
