"""
Ses Formatı Dönüştürücü

Vosk yalnızca Mono, 16kHz WAV formatını kabul eder.
Bu modül MP3, M4A, OGG gibi formatları Vosk'un beklediği
geçici WAV dosyasına çevirir.

Whisper yerel model tüm formatları doğrudan kabul ettiğinden
dönüştürme yalnızca Vosk için yapılır.

Araç: pydub + FFmpeg (sistem genelinde kurulu olmalı)
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Vosk'un gerektirdiği sabit değerler
VOSK_REQUIRED_CHANNELS    = 1       # Mono
VOSK_REQUIRED_SAMPLE_RATE = 16000   # 16 kHz
VOSK_REQUIRED_SAMPLE_WIDTH = 2      # 16-bit (2 byte)


def check_ffmpeg() -> None:
    """FFmpeg'in PATH'de erişilebilir olduğunu doğrular."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg bulunamadı. pydub MP3/M4A/OGG dönüşümü için FFmpeg gereklidir.\n"
            "Kurulum: https://ffmpeg.org/download.html\n"
            "Windows: 'winget install ffmpeg' veya PATH'e eklenmiş binary."
        )


def needs_conversion(audio_file_path: str) -> bool:
    """
    Dosyanın Vosk için dönüştürme gerektirip gerektirmediğini kontrol eder.
    WAV olmayan her format dönüştürme gerektirir.
    WAV olsa bile mono/16kHz değilse dönüştürme gerekir.
    """
    ext = Path(audio_file_path).suffix.lower()
    if ext != ".wav":
        return True

    # WAV ise kanal ve örnekleme hızını kontrol et
    import wave
    try:
        with wave.open(audio_file_path, "rb") as wf:
            return (
                wf.getnchannels()    != VOSK_REQUIRED_CHANNELS or
                wf.getframerate()    != VOSK_REQUIRED_SAMPLE_RATE or
                wf.getsampwidth()    != VOSK_REQUIRED_SAMPLE_WIDTH
            )
    except Exception:
        return True


def to_vosk_wav(audio_file_path: str) -> tuple[str, bool]:
    """
    Ses dosyasını Vosk'un kabul ettiği formata (Mono, 16kHz, 16-bit WAV) çevirir.

    Dönüştürme gereksizse orijinal dosya yolu döndürülür (kopyalama yapılmaz).
    Dönüştürme gerekiyorsa geçici bir WAV dosyası oluşturulur.

    Args:
        audio_file_path: Giriş ses dosyasının yolu.

    Returns:
        (wav_path, is_temp) çifti:
        - wav_path : Vosk için hazır WAV dosyasının yolu
        - is_temp  : True ise kullanım sonrası silinmeli (geçici dosya)

    Raises:
        FileNotFoundError : Giriş dosyası bulunamazsa.
        RuntimeError      : Dönüştürme başarısız olursa.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: '{audio_file_path}'")

    if not needs_conversion(audio_file_path):
        logger.debug("[DÖNÜŞTÜRÜCÜ] '%s' zaten uygun formatta.", audio_file_path)
        return audio_file_path, False

    logger.info(
        "[DÖNÜŞTÜRÜCÜ] '%s' → Vosk WAV formatına çevriliyor...",
        Path(audio_file_path).name,
    )

    check_ffmpeg()
    try:
        audio = AudioSegment.from_file(audio_file_path)
    except Exception as exc:
        raise RuntimeError(
            f"Ses dosyası okunamadı: '{audio_file_path}'\n"
            f"Hata: {exc}"
        ) from exc

    # Mono'ya çevir
    if audio.channels != VOSK_REQUIRED_CHANNELS:
        audio = audio.set_channels(VOSK_REQUIRED_CHANNELS)

    # 16 kHz'e çevir
    if audio.frame_rate != VOSK_REQUIRED_SAMPLE_RATE:
        audio = audio.set_frame_rate(VOSK_REQUIRED_SAMPLE_RATE)

    # 16-bit'e çevir
    if audio.sample_width != VOSK_REQUIRED_SAMPLE_WIDTH:
        audio = audio.set_sample_width(VOSK_REQUIRED_SAMPLE_WIDTH)

    # Geçici WAV dosyasına yaz (pipeline bitince silinir)
    tmp_file = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, prefix="voxsentinel_"
    )
    tmp_path = tmp_file.name
    tmp_file.close()

    audio.export(tmp_path, format="wav")
    logger.info("[DÖNÜŞTÜRÜCÜ] Geçici WAV oluşturuldu: '%s'", tmp_path)

    return tmp_path, True


def cleanup_temp_wav(wav_path: str, is_temp: bool) -> None:
    """
    to_vosk_wav() tarafından oluşturulan geçici dosyayı siler.
    is_temp=False ise hiçbir şey yapmaz (orijinal dosyaya dokunmaz).
    """
    if is_temp and os.path.exists(wav_path):
        os.remove(wav_path)
        logger.debug("[DÖNÜŞTÜRÜCÜ] Geçici dosya silindi: '%s'", wav_path)
