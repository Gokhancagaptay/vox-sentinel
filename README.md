# VoxSentinel

**English:** Multi-layer Turkish speech censorship for offline files and optional live microphone processing (Vosk + Whisper + phonetic voting + beep replacement).

Türkçe konuşma kayıtlarında yasaklı ifadeleri tespit edip ilgili zaman aralıklarına sansür (bip) uygular. Üç ana katman kullanır: **Vosk** (kelime zaman damgası), **Whisper** (API veya yerel model ile transkripsiyon) ve **fonetik benzerlik**; karar katmanında hizalama ve oylama, ses katmanında FFmpeg/pydub ile bip üretimi.

## Mimari (özet)

| Katman | Rol |
|--------|-----|
| ASR | Vosk zaman çizelgesi, Whisper metin/zaman, fonetik OOV yakalama |
| Karar | Zaman hizalama, çok kaynaklı oylama, segment birleştirme |
| Ses | pydub ile bip yerleştirme |

Ayrıntılı akış `main.py` dosya başlığındaki açıklamada ve `core/pipeline.py` içinde özetlenir.

## Gereksinimler

- **Python** 3.10 veya üzeri önerilir.
- **FFmpeg** sistemde kurulu ve `PATH` üzerinde erişilebilir olmalıdır ([indirme](https://ffmpeg.org/download.html)).
  - Windows: kurulum sonrası PATH’e ekleyin.
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
- **Vosk Türkçe modeli:** Depoda model dosyaları yoktur. [Vosk modelleri](https://alphacephei.com/vosk/models) sayfasından indirip `model/` içine açın; adımlar için [model/README.md](model/README.md).

## Kurulum

```bash
cd VoxSentinel
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- **Mikrofon aracı** (`mic_censor.py`): `pip install -r requirements-mic.txt`
- **Yardımcı scriptler** (`scripts/` altındaki denemeler, gTTS, SpeechRecognition): `pip install -r requirements-dev.txt`

`openai-whisper` PyTorch ile birlikte gelir; kurulum boyutu büyüktür. Yalnızca OpenAI Whisper API kullanacaksanız `config/settings.py` içinde `WHISPER_MODE = "api"` ayarlayıp API anahtarı tanımlayabilirsiniz.

## Ortam değişkenleri

| Değişken | Açıklama |
|----------|----------|
| `OPENAI_API_KEY` | Whisper **API** modu için gerekli. Tanımlı değilse `WHISPER_MODE="auto"` iken yerel Whisper veya yalnızca Vosk+fonetik devreye girer (`config/settings.py` mantığına bağlı). |

Anahtarları repoya eklemeyin; `.env` kullanıyorsanız `.gitignore` ile zaten dışlanır.

## Kullanım

**Dosya tabanlı pipeline (tam katman):**

```bash
python main.py giris.wav
python main.py giris.wav cikis.wav
```

**Mikrofon:**

```bash
python mic_censor.py
python mic_censor.py --stream
```

**Windows hızlı mikrofon testi:** `CIHAZ_SEC.bat` veya `python scripts\_mic_test_basit.py` (proje kökünden).

Diğer yardımcılar `scripts/` altında; çalıştırırken proje kökünden örnek:

```bash
python scripts\_cihaz_sec.py
python scripts\test_mic.py
```

`scripts/_bootstrap_path.py` proje kökünü `sys.path`’e ekler; `config` ve paket içe aktarımları bu sayede çalışır.

## Dizin yapısı

```
VoxSentinel/
├── main.py              # Dosya → pipeline
├── mic_censor.py        # Mikrofon / akış sansürü
├── requirements.txt
├── requirements-mic.txt
├── requirements-dev.txt
├── CIHAZ_SEC.bat
├── core/                # Orkestrasyon
├── asr/                 # Vosk, Whisper, fonetik
├── audio/               # Dönüştürme, bip
├── decision/            # Hizalama, oylama
├── config/              # Ayarlar, yasaklı/beyaz liste
├── scripts/             # Teşhis ve deneme araçları
└── model/               # README + sizin indirdiğiniz Vosk modeli (git’e girmez)
```

## Yasal ve etik uyarı

Otomatik sansür **yanlış pozitif/negatif** üretebilir. Bu yazılımı yalnızca yasalara ve kurum/kişi politikalarına uygun şekilde kullanın. Canlı ortamda kullanmadan önce kendi veri kümenizle doğrulama yapın.

## Lisans

Lisans bilgisi eklenmediyse depo sahibi tarafından belirlenmelidir.

## Uzak depo

```text
https://github.com/Gokhancagaptay/vox-sentinel.git
```
