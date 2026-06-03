"""
VoxSentinel — Çok Katmanlı Türkçe Ses Sansür Sistemi
=====================================================

Kullanım:
    python main.py <ses_dosyasi> [cikti_dosyasi] [--format wav|mp3|ogg]

Örnekler:
    python main.py sansur_test_sesi2.wav
    python main.py kayit.mp3 temiz_kayit.mp3 --format mp3

Ortam Değişkeni:
    OPENAI_API_KEY             → Whisper API kullanımı için gerekli.
    VOXSENTINEL_WHISPER_MODE   → api|local|auto (settings.py varsayılanını override eder)

Mimari (3 Katman):
    Katman 1  Vosk (zaman) + Whisper (transkripsiyon) + Fonetik (OOV)
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


def _print_startup_info(mode: str, whisper_source: str, model_size: str) -> None:
    """Başlangıçta kullanılan konfigürasyonu raporla."""
    from config.settings import VOSK_MODEL_PATH, PHONETIC_SIMILARITY_THRESHOLD
    from config.banned_words import YASAKLI_KELIMELER
    import shutil

    ffmpeg_ok = shutil.which("ffmpeg") is not None

    print(f"  Whisper modu      : {mode.upper()} → {whisper_source}")
    if whisper_source == "local":
        print(f"  Yerel model boyutu: {model_size}")
    print(f"  Vosk model yolu   : {VOSK_MODEL_PATH}")
    print(f"  Yasaklı kelimeler : {len(YASAKLI_KELIMELER)} adet")
    print(f"  Fonetik eşik      : {PHONETIC_SIMILARITY_THRESHOLD}")
    print(f"  FFmpeg            : {'OK' if ffmpeg_ok else 'BULUNAMADI!'}")
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
        from config.banned_words import agirlik_sayaci
        tum_tespitler = result.whisper_detections + result.phonetic_detections
        sayac = agirlik_sayaci(tum_tespitler)

        print()
        print(f"  Seviye dagılımı: Agir={sayac['yuksek']}  Orta={sayac['orta']}  Hafif={sayac['dusuk']}")
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
    # --format çıktı formatı (isteğe bağlı)
    output_format: str | None = None
    args = list(argv)
    if "--format" in args:
        fmt_idx = args.index("--format")
        if fmt_idx + 1 < len(args):
            output_format = args[fmt_idx + 1].lower().strip(".")
            del args[fmt_idx:fmt_idx + 2]
        else:
            print("HATA: --format bayrağından sonra format belirtilmedi (wav|mp3|ogg).")
            return 1

    if not args:
        print("Kullanım: python main.py <ses_dosyasi> [cikti_dosyasi] [--format wav|mp3|ogg]")
        print("Örnek   : python main.py sansur_test_sesi2.wav")
        print("Örnek   : python main.py kayit.mp3 temiz.mp3 --format mp3")
        return 1

    audio_file = args[0]
    if not Path(audio_file).exists():
        print(f"HATA: '{audio_file}' dosyası bulunamadı.")
        return 1

    # Çıkış dosyasını belirle
    if len(args) >= 2:
        output_file = args[1]
    else:
        ext = output_format or "wav"
        stem = Path(audio_file).stem
        output_file = f"{stem}_sansurlu.{ext}"

    # --format ile çıkış yolu uzantısı çelişiyorsa uzantıyı düzelt
    if output_format:
        out_path = Path(output_file)
        if out_path.suffix.lower().lstrip(".") != output_format:
            output_file = str(out_path.with_suffix(f".{output_format}"))

    # ── Whisper kaynak belirleme (pipeline.py mantığını burada tekrar çalıştırmıyoruz) ──
    from config.settings import WHISPER_MODE, WHISPER_LOCAL_MODEL_SIZE
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    use_whisper = True

    mode = WHISPER_MODE.lower()
    if mode == "api" and not has_openai_key:
        print("UYARI: WHISPER_MODE='api' ama OPENAI_API_KEY bulunamadi.")
        print("       Yerel mod devreye alınıyor.\n")
        whisper_source = "local"
    elif mode == "local" or (mode == "auto" and not has_openai_key):
        whisper_source = "local"
        print(f"[BİLGİ] Yerel Whisper aktif (model: {WHISPER_LOCAL_MODEL_SIZE})")
        if mode == "auto":
            print("        Model ilk çalıştırmada indirilecek.")
        print()
    else:
        whisper_source = "api"

    _print_startup_info(mode, whisper_source, WHISPER_LOCAL_MODEL_SIZE)

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
    except KeyboardInterrupt:
        print("\nİptal edildi.")
        return 1
    except Exception as exc:
        logger.exception("Beklenmeyen hata: %s", exc)
        return 1

    _print_result_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
