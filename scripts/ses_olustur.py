"""Google TTS ile test MP3 üretir (gTTS; requirements-dev.txt)."""
import _bootstrap_path  # noqa: F401

from gtts import gTTS

ROOT = _bootstrap_path.PROJECT_ROOT

metin = (
    "Merhaba. Bu çok sik ve piç bir test kaydıdır. "
    "Lütfen bu kötü kelimeleri sistemden sansürle."
)

print("Google TTS ile test sesi oluşturuluyor...")

tts = gTTS(text=metin, lang="tr")

dosya_adi = ROOT / "sansur_test_sesi2.mp3"
tts.save(str(dosya_adi))

print(f"Başarılı! '{dosya_adi}' oluşturuldu.")
