"""
VoxSentinel — Mikrofon Sansür Aracı
=====================================

Kullanım:
    python mic_censor.py              → Tam mod  (kaydet → işle → çal)
    python mic_censor.py --stream     → Akış modu (anlık, Vosk+Fonetik ~2s gecikme)
"""

import sys
import os
import io
import time
import wave
import json
import tempfile
import threading
import argparse
import subprocess
import queue
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sounddevice as sd
import numpy as np
import scipy.signal
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel

from config.settings import (
    VOSK_MODEL_PATH,
    BEEP_FREQUENCY_HZ, BEEP_GAIN_DB, CENSOR_PADDING_MS,
    MAX_RECORDING_DURATION_SEC,
)
from config.banned_words import YASAKLI_KELIMELER
from asr.phonetic_matcher import scan_for_phonetic_matches
from decision.voting_engine import vote_and_merge
from audio.censor_processor import apply_censor_beeps

# ── Sabitler ──────────────────────────────────────────────────────────────────
VOSK_RATE      = 16000      # Vosk'un beklediği örnekleme hızı
STREAM_SECONDS = 4          # Akış modunda her segment süresi
SILENCE_RMS    = 0.0005     # float32 RMS eşiği (SUNUM_DEMO referansıyla ayarlandı)
SILENCE_SECS   = 3.0        # Bu kadar sessizlik → kayıt durdurma

SetLogLevel(-1)


# ── Mikrofon tespiti ve seçimi ────────────────────────────────────────────────

def _mikrofon_sec() -> tuple[int, int, int]:
    """
    Çalışan tüm giriş cihazlarını listeler, kullanıcının seçmesini sağlar.
    Döndürür: (device_index, sample_rate, channels)
    """
    devices = sd.query_devices()
    hoparlor_anahtar = ("hoparlör", "speaker", "kulaklık", "headphone",
                        "output", "stereo mix", "karışım")

    print("\n  Calisabilir mikrofonlar aranıyor...")
    calisan: list[tuple[int, str, int, int]] = []   # (dev_id, ad, sr, ch)

    for dev_id, info in enumerate(devices):
        if info["max_input_channels"] == 0:
            continue

        ad_kucuk = info["name"].lower()
        if any(k in ad_kucuk for k in hoparlor_anahtar):
            continue

        sr = int(info["default_samplerate"])
        ch = min(info["max_input_channels"], 2)

        try:
            test = sd.rec(int(0.3 * sr), samplerate=sr, channels=ch,
                          device=dev_id, dtype="float32", blocking=True)
            vol = float(np.abs(test).mean())
            if vol > 1e10 or np.isnan(vol) or np.isinf(vol):
                continue
            calisan.append((dev_id, info["name"], sr, ch))
        except Exception:
            continue

    if not calisan:
        raise RuntimeError(
            "Hicbir calisan mikrofon bulunamadi!\n"
            "Discord/Teams/Zoom gibi uygulamalari kapatip tekrar deneyin."
        )

    # ── Listeyi göster ──────────────────────────────────────────────
    print(f"\n  {len(calisan)} calisan mikrofon bulundu:\n")
    for i, (dev_id, ad, sr, ch) in enumerate(calisan):
        isaretci = " <-- ilk secim" if i == 0 else ""
        print(f"    [{dev_id:2d}] {ad:<45} (sr={sr} Hz){isaretci}")

    print(f"\n  Enter  = ilk secim ([{calisan[0][0]}] {calisan[0][1]})")
    print("  Numara = baska bir cihaz index'i girin")
    secim = input("\n  Seciminiz: ").strip()

    if secim == "":
        dev_id, ad, sr, ch = calisan[0]
    else:
        try:
            istenen = int(secim)
            eslesen = [(d, a, s, c) for d, a, s, c in calisan if d == istenen]
            if eslesen:
                dev_id, ad, sr, ch = eslesen[0]
            else:
                print(f"  [{istenen}] calisan listede yok, ilk secim kullanılıyor.")
                dev_id, ad, sr, ch = calisan[0]
        except ValueError:
            dev_id, ad, sr, ch = calisan[0]

    print(f"\n  Secilen: [{dev_id}] {ad}  (sr={sr} Hz, ch={ch})\n")
    return dev_id, sr, ch


# ── Ses dönüşümü: float32 native → int16 16kHz (Vosk için) ──────────────────

def _donustur_vosk_wav(float32_array: np.ndarray, native_sr: int, path: str) -> None:
    """
    sounddevice'dan gelen float32 veriyi Vosk'un beklediği
    mono 16kHz int16 WAV formatına dönüştürür.
    """
    # Stereo → mono
    if float32_array.ndim > 1:
        mono = float32_array.mean(axis=1)
    else:
        mono = float32_array.flatten()

    # Yeniden örnekleme (native → 16kHz)
    if native_sr != VOSK_RATE:
        num_out = int(len(mono) * VOSK_RATE / native_sr)
        mono = scipy.signal.resample(mono, num_out)

    # float32 → int16
    mono_int16 = np.clip(mono * 32767, -32768, 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(VOSK_RATE)
        wf.writeframes(mono_int16.tobytes())


# ── Oynatma ───────────────────────────────────────────────────────────────────

def _ses_cal(dosya: str) -> None:
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", dosya],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TAM MOD — Kaydet → Tam Pipeline → Çal
# ═══════════════════════════════════════════════════════════════════════════════

def tam_mod():
    from core.pipeline import run_censorship_pipeline

    print("\n" + "=" * 58)
    print("  TAM MOD — Kaydet -> Isle -> Cal")
    print("=" * 58)
    print("  Konusmaya baslayin.")
    print("  Durmak icin  -> Enter'a basin")
    print(f"  Veya {SILENCE_SECS:.0f}s sessizlik -> otomatik durur")
    print(f"  Maksimum kayit suresi: {MAX_RECORDING_DURATION_SEC}s")
    print("=" * 58 + "\n")

    try:
        dev_id, native_sr, ch = _mikrofon_sec()
    except RuntimeError as e:
        print(f"  HATA: {e}")
        return

    print(f"  Kayit rate : {native_sr} Hz (Vosk icin {VOSK_RATE} Hz'e donusturulecek)\n")

    # Callback tabanlı kayıt kuyruğu (WDM-KS blocking desteklemiyor)
    blok_kuyrugu: queue.Queue = queue.Queue()

    def ses_callback(indata, frames, zaman, durum):
        blok_kuyrugu.put(indata.copy())

    kayit_aktif = threading.Event()
    kayit_aktif.set()
    kareler: list[np.ndarray] = []
    sessizlik_sn = 0.0
    chunk_sn = 1024 / native_sr

    def enter_bekle():
        input()
        kayit_aktif.clear()

    threading.Thread(target=enter_bekle, daemon=True).start()
    print("  [KAYIT] Basladi... (Enter ile dur)\n")
    baslangic = time.time()

    print("  (Mikrofon seviyesi gercek zamanli gosteriliyor — konusmaya baslayin)\n")

    with sd.InputStream(samplerate=native_sr, channels=ch, dtype="float32",
                        device=dev_id, blocksize=1024, callback=ses_callback):
        while kayit_aktif.is_set():
            try:
                blok = blok_kuyrugu.get(timeout=0.5)
            except queue.Empty:
                continue

            kareler.append(blok)
            seviye = float(np.abs(blok).mean())

            if seviye < SILENCE_RMS:
                sessizlik_sn += chunk_sn
            else:
                sessizlik_sn = 0.0

            # Her durumda anlık seviyeyi göster
            elapsed  = time.time() - baslangic
            bar_len  = min(int(seviye * 1000), 30)
            durum    = "SESSIZ" if seviye < SILENCE_RMS else "SES   "
            cubuk    = "#" * bar_len + "-" * (30 - bar_len)
            kalan_ss = max(0.0, SILENCE_SECS - sessizlik_sn)
            sys.stdout.write(
                f"\r  [{durum}] {elapsed:.1f}s  |{cubuk}|  rms={seviye:.5f}"
                f"  sessizlik:{kalan_ss:.1f}s/{SILENCE_SECS:.0f}s"
            )
            sys.stdout.flush()

            if sessizlik_sn >= SILENCE_SECS:
                print(f"\n  [{SILENCE_SECS:.0f}s sessizlik] Kayit durdu.")
                break

            if time.time() - baslangic >= MAX_RECORDING_DURATION_SEC:
                print(f"\n  [MAX SURE] {MAX_RECORDING_DURATION_SEC}s limitine ulasildi. Kayit durdu.")
                break

    if not kareler:
        print("  Ses kaydedilemedi.")
        return

    ses_verisi = np.concatenate(kareler, axis=0)
    sure = len(ses_verisi) / native_sr
    print(f"\n  Toplam kayit: {sure:.1f}s")

    if sure < 1.0:
        print("  Cok kisa kayit, tekrar deneyin.")
        return

    # Vosk formatına dönüştür ve kaydet
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="vox_mic_")
    tmp.close()
    tmp_path = tmp.name
    cikti = Path(tmp_path).stem + "_sansurlu.wav"

    try:
        _donustur_vosk_wav(ses_verisi, native_sr, tmp_path)

        print("\n  [ISLEM] Whisper + Vosk + Fonetik basliyor...")
        t0 = time.time()

        sonuc = run_censorship_pipeline(
            audio_file_path=tmp_path,
            output_path=cikti,
            use_vosk=True,
            use_whisper=True,
            use_phonetic=True,
        )

        print(f"  Tamamlandi ({time.time()-t0:.1f}s)")
        # ── Transkripsiyon raporu ────────────────────────────────────────
        print("\n" + "=" * 58)
        print("  TRANSKRIPSIYON RAPORU")
        print("=" * 58)

        # Hangi kelimelerin sansürlendiğini bul (zaman aralığına göre)
        def _sansur_mu(start_s: float, end_s: float) -> str | None:
            """Kelime zaman aralığı bir sansür segmentiyle örtüşüyor mu?"""
            start_ms = int(start_s * 1000)
            end_ms   = int(end_s   * 1000)
            for seg in sonuc.final_censor_segments:
                if start_ms < seg["end_ms"] and end_ms > seg["start_ms"]:
                    return seg.get("matched_banned", "?")
            return None

        # Whisper transkripsiyon satırı
        if sonuc.whisper_words:
            print("\n  Whisper (tam metin):")
            tam_metin = " ".join(w["word"] for w in sonuc.whisper_words)
            print(f"    {tam_metin}\n")

            print("  Kelime detayi  (S=sansurlu, T=temiz):")
            print(f"  {'Kelime':<20} {'Baslangic':>10} {'Bitis':>10}  Durum")
            print("  " + "-" * 52)
            for w in sonuc.whisper_words:
                eslesme = _sansur_mu(w["start"], w["end"])
                if eslesme:
                    durum = f"[S] -> '{eslesme}' biplenecek"
                else:
                    durum = "[T] temiz"
                print(f"  {w['word']:<20} {w['start']:>9.2f}s {w['end']:>9.2f}s  {durum}")
        else:
            print("\n  Whisper cikti uretmedi (cok kisa ses?).")

        # Vosk özeti
        if sonuc.vosk_words:
            vosk_metin = " ".join(w["word"] for w in sonuc.vosk_words)
            print(f"\n  Vosk (timestamp ref): {vosk_metin}")

        # Fonetik tespitler
        if sonuc.phonetic_detections:
            print(f"\n  Fonetik eslesme ({len(sonuc.phonetic_detections)} adet):")
            for d in sonuc.phonetic_detections:
                print(f"    '{d['word']}' ~ '{d.get('matched_banned','?')}'"
                      f"  (skor={d.get('score',0):.2f})")

        # Final özet
        print("\n" + "=" * 58)
        if sonuc.final_censor_segments:
            print(f"  SONUC: {len(sonuc.final_censor_segments)} kelime biplendi\n")
            for seg in sonuc.final_censor_segments:
                print(f"    [{seg['source']:>22}]  '{seg.get('word','?')}'"
                      f" -> '{seg.get('matched_banned','?')}'"
                      f"  ({seg['start_ms']}ms - {seg['end_ms']}ms)")
            print(f"\n  Sansurlu dosya: {cikti}")
            print("\n  Oynatiliyor...\n")
            _ses_cal(cikti)
        else:
            print("  SONUC: Kufur tespit edilmedi — temiz kayit.")
            print("\n  Oynatiliyor...\n")
            _ses_cal(tmp_path)

        print("=" * 58)

    finally:
        # Temp dosyaları her durumda temizle (hata olsa bile)
        for f in [tmp_path, cikti]:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# AKIŞ MODU — Anlık chunk (Vosk + Fonetik, ~2s gecikme)
# ═══════════════════════════════════════════════════════════════════════════════

def akis_modu():
    print("\n" + "=" * 58)
    print("  AKIS MODU — Anlik Sansur (~2s gecikme)")
    print("=" * 58)
    print("  Konusmaya baslayin. Kufurler bip ile kapatilacak.")
    print("  Cikmak icin Ctrl+C")
    print("=" * 58 + "\n")

    try:
        dev_id, native_sr, ch = _mikrofon_sec()
    except RuntimeError as e:
        print(f"  HATA: {e}")
        return

    try:
        model = Model(VOSK_MODEL_PATH)
    except Exception as exc:
        print(f"  HATA: Vosk modeli yuklenemedi: {exc}")
        print(f"  Model dizini: {VOSK_MODEL_PATH}")
        return
    ses_kuyruğu: queue.Queue = queue.Queue()
    biriken: list[np.ndarray] = []
    kare_limiti = int(native_sr * STREAM_SECONDS / 1024)
    segment_no = 0

    def ses_callback(indata, frames, zaman, durum):
        ses_kuyruğu.put(indata.copy())

    print("  [AKIS] Dinleniyor...\n")
    try:
        with sd.InputStream(samplerate=native_sr, channels=ch, dtype="float32",
                            blocksize=1024, device=dev_id, callback=ses_callback):
            while True:
                blok = ses_kuyruğu.get()
                biriken.append(blok)
                if len(biriken) >= kare_limiti:
                    _islem_chunk(biriken, segment_no, model, native_sr)
                    segment_no += 1
                    biriken.clear()
    except KeyboardInterrupt:
        print("\n\n  [AKIS] Durduruldu.")


def _islem_chunk(kareler: list[np.ndarray], idx: int, model, native_sr: int) -> None:
    """Bir segment'i Vosk + Fonetik ile işler, sansürlü halini çalar."""
    ses = np.concatenate(kareler, axis=0)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"vox_chunk{idx}_")
    tmp.close()
    tmp_path = tmp.name
    cikti = tmp_path.replace(".wav", "_sansur.wav")

    try:
        _donustur_vosk_wav(ses, native_sr, tmp_path)

        # Vosk transkripsiyon
        rec = KaldiRecognizer(model, VOSK_RATE)
        rec.SetWords(True)
        with wave.open(tmp_path, "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                rec.AcceptWaveform(data)

        try:
            kelimeler = json.loads(rec.FinalResult()).get("result", [])
        except json.JSONDecodeError:
            kelimeler = []

        if not kelimeler:
            print(f"  [CHUNK {idx}] Sessiz / Anlasılamadı")
            return

        fonetik = scan_for_phonetic_matches(kelimeler)

        vosk_direkt = []
        for w in kelimeler:
            for banned in YASAKLI_KELIMELER:
                if banned.lower() in w["word"].lower():
                    vosk_direkt.append({
                        "word": w["word"], "start": w["start"], "end": w["end"],
                        "matched_banned": banned, "source": "vosk-direct"
                    })
                    break

        segmentler = vote_and_merge(vosk_direkt, fonetik)

        if segmentler:
            apply_censor_beeps(tmp_path, segmentler, cikti)
            bulunanlar = [f"'{s.get('word','?')}'" for s in segmentler]
            print(f"  [CHUNK {idx}] TESPIT: {', '.join(bulunanlar)} → biplendi")
            _ses_cal(cikti)
        else:
            metin = " ".join(w["word"] for w in kelimeler)
            print(f"  [CHUNK {idx}] Temiz: '{metin}'")
            _ses_cal(tmp_path)

    finally:
        for f in [tmp_path, cikti]:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# GİRİŞ NOKTASI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxSentinel — Mikrofon Sansur Araci")
    parser.add_argument("--stream", action="store_true",
                        help="Akis modu: ~2s gecikme (Vosk+Fonetik)")
    args = parser.parse_args()

    if args.stream:
        akis_modu()
    else:
        tam_mod()
