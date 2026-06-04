"""Vosk kelime zamanları + pydub bip ile basit dosya sansürü (tek katmanlı örnek)."""
import json
import sys
import wave

import _bootstrap_path  # noqa: F401
from pydub import AudioSegment
from pydub.generators import Sine
from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)

ROOT = _bootstrap_path.PROJECT_ROOT
ses_dosyasi = ROOT / "sansur_test_sesi2.wav"
cikti_dosyasi = ROOT / "sansurlu_sonuc2.wav"

print("\n[YÜKLENİYOR] Yapay zeka ve ses motorları başlatılıyor...")
model = Model(str(ROOT / "model"))

try:
    wf = wave.open(str(ses_dosyasi), "rb")
    orijinal_ses = AudioSegment.from_wav(str(ses_dosyasi))
except FileNotFoundError:
    print(f"\nHATA: '{ses_dosyasi}' bulunamadı!")
    sys.exit(1)

recognizer = KaldiRecognizer(model, wf.getframerate())
recognizer.SetWords(True)

yasakli_kelimeler = ["aptal", "salak", "kötü,sik,piç"]

print(f"\n[ANALİZ BAŞLADI] '{ses_dosyasi}' taranıyor...\n")
print("=" * 60)

sansurlenecek_zamanlar = []
tam_metin = ""

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break

    if recognizer.AcceptWaveform(data):
        sonuc = json.loads(recognizer.Result())
        if "result" in sonuc:
            for kelime_bilgisi in sonuc["result"]:
                kelime = kelime_bilgisi["word"].lower()
                tam_metin += kelime + " "

                for yasakli in yasakli_kelimeler:
                    if yasakli in kelime:
                        start_ms = int(kelime_bilgisi["start"] * 1000)
                        end_ms = int(kelime_bilgisi["end"] * 1000)
                        sansurlenecek_zamanlar.append((start_ms, end_ms, kelime))
                        print(
                            f"[TESPIT] '{kelime.upper()}' bulundu! "
                            f"(Zaman: {kelime_bilgisi['start']}s - {kelime_bilgisi['end']}s)"
                        )

son_sonuc = json.loads(recognizer.FinalResult())
if "result" in son_sonuc:
    for kelime_bilgisi in son_sonuc["result"]:
        kelime = kelime_bilgisi["word"].lower()
        tam_metin += kelime + " "
        for yasakli in yasakli_kelimeler:
            if yasakli in kelime:
                start_ms = int(kelime_bilgisi["start"] * 1000)
                end_ms = int(kelime_bilgisi["end"] * 1000)
                sansurlenecek_zamanlar.append((start_ms, end_ms, kelime))
                print(
                    f"[TESPIT] '{kelime.upper()}' bulundu! "
                    f"(Zaman: {kelime_bilgisi['start']}s - {kelime_bilgisi['end']}s)"
                )

print("=" * 60)

if sansurlenecek_zamanlar:
    print("\n[SANSÜR İŞLEMİ] Ses dosyasına bip uygulanıyor...")

    sansurlenecek_zamanlar.sort(key=lambda x: x[0], reverse=True)

    for start, end, _kelime in sansurlenecek_zamanlar:
        sure = end - start
        bip_sesi = Sine(1000).to_audio_segment(duration=sure).apply_gain(-5)
        orijinal_ses = orijinal_ses[:start] + bip_sesi + orijinal_ses[end:]

    orijinal_ses.export(str(cikti_dosyasi), format="wav")
    print(f"\n[OK] Yeni dosya kaydedildi: '{cikti_dosyasi}'")
else:
    print("\n[OK] Temiz dosya. Sansürlenecek kelime bulunamadı.")

print(f"\nOrijinal Metin Dökümü: {tam_metin}")
