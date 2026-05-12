"""Vosk ile WAV dosyası üzerinde basit kelime taraması."""
import _bootstrap_path  # noqa: F401

import wave
import json
import sys

from vosk import Model, KaldiRecognizer

ROOT = _bootstrap_path.PROJECT_ROOT
ses_dosyasi = ROOT / "test_sesi.wav"

print("Yapay zeka modeli yükleniyor...")
model = Model(str(ROOT / "model"))

try:
    wf = wave.open(str(ses_dosyasi), "rb")
except FileNotFoundError:
    print(f"\nHATA: '{ses_dosyasi}' adında bir dosya bulunamadı!")
    print("Lütfen proje köküne bir test_sesi.wav ekleyin veya yolu güncelleyin.")
    sys.exit(1)

if wf.getnchannels() != 1:
    print("\nHATA: Ses dosyası 'Mono' (tek kanal) olmalıdır. Stereo ses desteklenmez.")
    sys.exit(1)

recognizer = KaldiRecognizer(model, wf.getframerate())

hedef_kelimeler = ["proje", "test", "gökhan"]

print(f"\n'{ses_dosyasi}' analiz ediliyor...\n")
print("-" * 50)

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break

    if recognizer.AcceptWaveform(data):
        sonuc = json.loads(recognizer.Result())
        metin = sonuc.get("text", "")

        if metin:
            print(f"Metin: {metin}")

            for kelime in hedef_kelimeler:
                if kelime in metin.lower():
                    print("\n" + "!" * 40)
                    print(f"  HEDEF YAKALANDI: {kelime.upper()}")
                    print("!" * 40 + "\n")

son_sonuc = json.loads(recognizer.FinalResult())
son_metin = son_sonuc.get("text", "")
if son_metin:
    print(f"Metin: {son_metin}")
    for kelime in hedef_kelimeler:
        if kelime in son_metin.lower():
            print("\n" + "!" * 40)
            print(f"  HEDEF YAKALANDI: {kelime.upper()}")
            print("!" * 40 + "\n")

print("-" * 50)
print("Analiz tamamlandı.")
