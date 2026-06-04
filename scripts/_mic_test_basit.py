"""Mikrofon izni verildi mi? Kısa kayıt testi."""
import io
import sys

import _bootstrap_path  # noqa: F401

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import sounddevice as sd

SURE = 2  # saniye

print("Mevcut giriş cihazları taranıyor...\n")
devices = sd.query_devices()
calisan = []

for dev_id, info in enumerate(devices):
    if info["max_input_channels"] == 0:
        continue

    sr = int(info["default_samplerate"])
    ch = min(info["max_input_channels"], 2)
    ad = info["name"]

    try:
        test = sd.rec(
            int(0.3 * sr),
            samplerate=sr,
            channels=ch,
            device=dev_id,
            dtype="float32",
            blocking=True,
        )
        vol = float(np.abs(test).mean())
        if vol > 1e10 or np.isnan(vol):
            raise ValueError("Geçersiz ses seviyesi")
        calisan.append((dev_id, ad, sr, ch))
        print(f"  [OK] [{dev_id:2d}] {ad}  (sr={sr}, ch={ch})")
    except Exception as e:
        hata = str(e).split("\n")[0][:60]
        print(f"  [--] [{dev_id:2d}] {ad}  → {hata}")

print()
if not calisan:
    print("Hicbir cihaz calismadi.")
else:
    ilk = calisan[0]
    print(f"Test kaydı: [{ilk[0]}] {ilk[1]}  ({SURE}s, konuşun...)")
    data = sd.rec(
        int(SURE * ilk[2]),
        samplerate=ilk[2],
        channels=ilk[3],
        device=ilk[0],
        dtype="float32",
        blocking=True,
    )
    rms = float(np.sqrt(np.mean(data**2)))
    print(f"Kayit tamam!  RMS={rms:.5f}  ({'SES VAR' if rms > 0.001 else 'sessiz'})")
    print(f"\nONERILEN CİHAZ INDEX: {ilk[0]}")
