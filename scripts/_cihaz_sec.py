"""
Mikrofon cihazlarını listeler, kullanıcının seçtiği cihazla kayıt testi yapar.
Çalışan cihaz bulunursa önerilen index'i köke `mic_device_config.py` olarak kaydedebilir.

Çalıştırma (proje kökünden):
    python scripts/_cihaz_sec.py
"""
import _bootstrap_path  # noqa: F401
from pathlib import Path

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pyaudio
import wave
import struct

ROOT = _bootstrap_path.PROJECT_ROOT

SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024
TEST_SURE = 3  # saniye

p = pyaudio.PyAudio()

# ── Giriş cihazlarını listele ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Mevcut Mikrofon / Giriş Cihazları")
print("=" * 60)

giris_cihazlari = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        host = p.get_host_api_info_by_index(info["hostApi"])["name"]
        giris_cihazlari.append((i, info["name"], host, int(info["defaultSampleRate"])))
        print(f"  [{i:2d}]  {info['name']:<45}  ({host})")

print("=" * 60)

# ── Kullanıcıdan seçim al ─────────────────────────────────────────────────
print("\nHangi cihazı denemek istiyorsunuz? Index numarasını girin")
print("(Birden fazla denemek için virgülle ayırın, örn: 1,2,14)")
print("Hepsini otomatik dene için 'hepsi' yazın\n")

girdi = input("Seçim: ").strip().lower()

if girdi == "hepsi":
    secilen = [c[0] for c in giris_cihazlari]
else:
    try:
        secilen = [int(x.strip()) for x in girdi.split(",")]
    except ValueError:
        print("Geçersiz giriş. Çıkılıyor.")
        p.terminate()
        sys.exit(1)

# ── Seçilen cihazları test et ─────────────────────────────────────────────
print(f"\n{TEST_SURE} saniyelik test kaydı yapılacak — konuşun!\n")

calisan_cihazlar = []

for dev_idx in secilen:
    info = p.get_device_info_by_index(dev_idx)
    cihaz_adi = info["name"]
    native_rate = int(info["defaultSampleRate"])

    print(f"  Deneniyor [{dev_idx:2d}] {cihaz_adi}  (native {native_rate} Hz)...")

    for rate in sorted(set([native_rate, 16000, 44100, 48000])):
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=rate,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=CHUNK,
            )

            kareler = []
            for _ in range(int(rate / CHUNK * TEST_SURE)):
                veri = stream.read(CHUNK, exception_on_overflow=False)
                kareler.append(veri)

            stream.stop_stream()
            stream.close()

            tum_veri = b"".join(kareler)
            ornekler = struct.unpack(f"{len(tum_veri)//2}h", tum_veri)
            rms = (sum(s * s for s in ornekler) / len(ornekler)) ** 0.5

            print(
                f"    [OK] rate={rate} Hz — RMS={rms:.1f} "
                f"({'ses alindi!' if rms > 50 else 'sessiz ama akis acildi'})"
            )

            kayit_yolu = ROOT / f"_test_{dev_idx}_{rate}.wav"
            with wave.open(str(kayit_yolu), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(tum_veri)
            print(f"    Kayit: {kayit_yolu}")

            calisan_cihazlar.append((dev_idx, rate, cihaz_adi, rms))
            break

        except Exception as e:
            print(f"    [--] rate={rate} Hz — {e}")

    print()

p.terminate()

# ── Özet ──────────────────────────────────────────────────────────────────
print("=" * 60)
if calisan_cihazlar:
    print("  CALISAN CIHAZLAR:\n")
    for idx, rate, ad, rms in calisan_cihazlar:
        print(f"  [{idx:2d}] {ad}  @ {rate} Hz  (RMS={rms:.1f})")

    en_iyi = max(calisan_cihazlar, key=lambda x: x[3])
    print(f"\n  ONERI: index={en_iyi[0]}, rate={en_iyi[1]}")
    print(f"  '{en_iyi[2]}'")

    cevap = input("\n  Bu cihazı köke mic_device_config.py olarak kaydetmek ister misiniz? (e/h): ").strip().lower()
    if cevap == "e":
        ayar_dosyasi = ROOT / "mic_device_config.py"
        with open(ayar_dosyasi, "w", encoding="utf-8") as f:
            f.write("# Otomatik tespit edildi\n")
            f.write(f"MIC_DEVICE_INDEX = {en_iyi[0]}\n")
            f.write(f"MIC_SAMPLE_RATE  = {en_iyi[1]}\n")
            f.write(f'MIC_DEVICE_NAME  = "{en_iyi[2]}"\n')
        print(f"  Kaydedildi: {ayar_dosyasi}")
else:
    print("  Hicbir cihaz calismadi.")
    print("  Oneril: Cursor disinda normal PowerShell ile tekrar deneyin.")
print("=" * 60)
