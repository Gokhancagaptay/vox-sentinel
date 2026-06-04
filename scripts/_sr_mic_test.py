"""SpeechRecognition kütüphanesi ile mikrofon erişim testi."""
import io
import sys

import _bootstrap_path  # noqa: F401

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import speech_recognition as sr

r = sr.Recognizer()

print("Mevcut mikrofonlar:")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"  [{i:2d}] {name}")

print("\n2 saniye kayıt yapılıyor (Huaxu-X1 denenecek), konuşun...")

for mic_index in [1, 2, None]:
    label = f"index={mic_index}" if mic_index is not None else "varsayılan"
    try:
        with sr.Microphone(device_index=mic_index, sample_rate=16000) as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.record(source, duration=1.5)
        raw = audio.get_raw_data()
        print(f"[{label}] OK — {len(raw)} byte kayıt alındı")
        break
    except Exception as e:
        print(f"[{label}] FAIL — {e}")
