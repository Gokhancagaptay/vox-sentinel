"""Çalışan mikrofon cihazını tespit eder (WMME + WASAPI dener)."""
import _bootstrap_path  # noqa: F401

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyaudio

FORMAT = pyaudio.paInt16
CHUNK = 1024
DENEME_RATELARI = [16000, 44100, 48000, 22050, 8000]

p = pyaudio.PyAudio()

print("Host API'ler:")
wasapi_index = None
for hi in range(p.get_host_api_count()):
    h = p.get_host_api_info_by_index(hi)
    print(f"  [{hi}] {h['name']}  (type={h['type']})")
    if "WASAPI" in h["name"]:
        wasapi_index = hi
print()

calisan = []


def test_device(dev_idx, rate, extra_kwargs=None):
    kwargs = dict(
        format=FORMAT,
        channels=1,
        rate=rate,
        input=True,
        input_device_index=dev_idx,
        frames_per_buffer=CHUNK,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    try:
        stream = p.open(**kwargs)
        stream.read(CHUNK, exception_on_overflow=False)
        stream.stop_stream()
        stream.close()
        return True
    except Exception:
        return False


print("Çalışan mikrofon cihazı aranıyor...\n")

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] < 1:
        continue

    host_name = p.get_host_api_info_by_index(info["hostApi"])["name"]
    basarili_rate = None

    for rate in DENEME_RATELARI:
        if test_device(i, rate):
            basarili_rate = rate
            break

    if basarili_rate:
        calisan.append((i, basarili_rate))
        print(f"  [OK] [{i:2d}] {info['name']}  (host={host_name}, rate={basarili_rate})")
    else:
        print(f"  [--] [{i:2d}] {info['name']}  (host={host_name})  → tüm rate'ler başarısız")

p.terminate()

print()
if calisan:
    best_idx, best_rate = calisan[0]
    print(f"ÖNERİLEN CİHAZ: index={best_idx}, rate={best_rate}")
    print("mic_censor.py başına şunu ekleyin (veya settings.py'e):")
    print(f"  MIC_DEVICE_INDEX = {best_idx}")
    print(f"  MIC_SAMPLE_RATE  = {best_rate}")
else:
    print("Hiçbir çalışan mikrofon bulunamadı!")
    print("Çözüm önerileri:")
    print("  1) Windows Gizlilik Ayarları → Mikrofona erişime izin ver")
    print("  2) Ses sürücüsünü güncelle / yeniden yükle")
    print("  3) Farklı bir USB ses kartı dene")
    print("  4) sounddevice kütüphanesiyle tekrar dene: pip install sounddevice")
