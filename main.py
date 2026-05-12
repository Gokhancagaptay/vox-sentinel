"""
VoxSentinel — Çok Katmanlı Türkçe Ses Sansür Sistemi
=====================================================

Kullanım:
    python main.py <ses_dosyasi> [cikti_dosyasi]

Örnekler:
    python main.py sansur_test_sesi2.wav
    python main.py kayit.mp3 temiz_kayit.wav

Ortam Değişkeni:
    OPENAI_API_KEY  → Whisper API kullanımı için gerekli.
                      Yoksa yalnızca Vosk + fonetik katman çalışır.

Mimari (3 Katman):
    Katman 1  Vosk (zaman) + Whisper API (transkripsiyon) + Fonetik (OOV)
    Katman 2  Zaman hizalama + Çok-kaynaklı oylama
    Katman 3  FFmpeg/pydub bip ekleme
"""

import sys
import os
import logging
import io
from pathlib import Path

# Windows terminalinde UTF-8 çıktısı için stdout'u yeniden yapılandır
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from core.pipeline import run_censorship_pipeline

# ─── Loglama yapılandırması ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _print_banner() -> None:
    print()
    print("=" * 60)
    print("  VoxSentinel --- Cok Katmanli Ses Sansur Sistemi")
    print("=" * 60)
    print()


def _print_result_summary(result) -> None:
    print()
    print("-" * 60)
    print("  SONUC RAPORU")
    print("-" * 60)
    print(f"  Vosk kelime sayisi       : {len(result.vosk_words)}")
    print(f"  Whisper kelime sayisi    : {len(result.whisper_words)}")
    print(f"  Anchor nokta sayisi      : {result.anchor_count}")
    print(f"  Whisper tespitleri       : {len(result.whisper_detections)}")
    print(f"  Fonetik tespitler        : {len(result.phonetic_detections)}")
    print(f"  Final sansur segmentleri : {len(result.final_censor_segments)}")
    print("-" * 60)

    if result.final_censor_segments:
        print()
        print("  Sansürlenen Kelimeler:")
        for seg in result.final_censor_segments:
            print(
                f"    [{seg['source']:>15}] '{seg.get('word', '?')}'"
                f" -> {seg['matched_banned']}"
                f" ({seg['start_ms']}ms - {seg['end_ms']}ms)"
            )
        print()
        print(f"  [OK] Cikis dosyasi : {result.output_file}")
    else:
        print()
        print("  [OK] Temiz dosya - Sansurlenecek kelime bulunamadi.")

    print("-" * 60)
    print()


def main(argv: list[str] = sys.argv[1:]) -> int:
    """
    Ana giriş noktası.

    Returns:
        0 başarı, 1 hata için çıkış kodu.
    """
    _print_banner()

    # ── Argüman işleme ────────────────────────────────────────────
    if not argv:
        print("Kullanım: python main.py <ses_dosyasi> [cikti_dosyasi]")
        print("Örnek   : python main.py sansur_test_sesi2.wav")
        return 1

    audio_file = argv[0]
    if not Path(audio_file).exists():
        print(f"HATA: '{audio_file}' dosyası bulunamadı.")
        return 1

    # Çıkış dosyasını belirle: verilmezse girişin yanına "_sansurlu" ekle
    if len(argv) >= 2:
        output_file = argv[1]
    else:
        stem = Path(audio_file).stem
        output_file = f"{stem}_sansurlu.wav"

    # ── Whisper mod belirleme ─────────────────────────────────────
    from config.settings import WHISPER_MODE, WHISPER_LOCAL_MODEL_SIZE
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    use_whisper = True  # Yerel mod her zaman kullanılabilir

    mode = WHISPER_MODE.lower()
    if mode == "api" and not has_openai_key:
        print("UYARI: WHISPER_MODE='api' ama OPENAI_API_KEY bulunamadi.")
        print("       settings.py'de WHISPER_MODE='local' yapın veya API anahtarı girin.\n")
        use_whisper = False
    elif mode == "auto" and not has_openai_key:
        print(f"[BİLGİ] API anahtarı yok → Yerel Whisper (model: {WHISPER_LOCAL_MODEL_SIZE}) kullanılıyor.")
        print("        Model ilk çalıştırmada indirilecek (~150MB base için).\n")
    elif mode == "local" or (mode == "auto" and not has_openai_key):
        print(f"[BİLGİ] Yerel Whisper aktif (model: {WHISPER_LOCAL_MODEL_SIZE})\n")

    # ── Boru hattını çalıştır ────────────────────────────────────
    try:
        result = run_censorship_pipeline(
            audio_file_path=audio_file,
            output_path=output_file,
            use_vosk=True,
            use_whisper=use_whisper,
            use_phonetic=True,
        )
    except FileNotFoundError as exc:
        print(f"HATA: {exc}")
        return 1
    except Exception as exc:
        logger.exception("Beklenmeyen hata: %s", exc)
        return 1

    _print_result_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
