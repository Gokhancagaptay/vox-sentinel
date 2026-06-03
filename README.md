# VoxSentinel

**English:** Multi-layer Turkish speech censorship for offline files and live microphone processing (Vosk + Whisper + phonetic voting + beep replacement).

Türkçe konuşma kayıtlarında yasaklı ifadeleri tespit edip ilgili zaman aralıklarına sansür (bip) uygular. Üç katman kullanır: **Vosk** (kelime zaman damgası), **Whisper** (API veya yerel model, transkripsiyon) ve **fonetik benzerlik** (Jaro-Winkler); karar katmanında DP-LCS tabanlı zaman hizalama ve çok-kaynaklı oylama, ses katmanında FFmpeg/pydub ile bip üretimi.

## Mimari

```
Katman 1 — Paralel ASR
  ├─ Vosk          → kelime zaman damgaları (zaman makinesi)
  ├─ Whisper       → doğru transkripsiyon + zaman damgaları
  └─ Fonetik       → Vosk OOV tespiti (Jaro-Winkler fuzzy match)

Katman 2 — Karar Motoru
  ├─ Zaman hizalama  → DP-LCS anchor bulma, Vosk ts → Whisper ts
  └─ Çok-kaynaklı oy → Whisper ∨ Fonetik → sansür segmenti

Katman 3 — Ses İşlemi
  └─ O(n) tek geçişli bip yerleştirme (pydub + FFmpeg)
```

| Modül | Dosya |
|-------|-------|
| ASR | `asr/vosk_engine.py`, `asr/whisper_engine.py`, `asr/whisper_local_engine.py`, `asr/phonetic_matcher.py` |
| Karar | `decision/time_aligner.py`, `decision/voting_engine.py` |
| Ses | `audio/censor_processor.py`, `audio/converter.py` |
| Orkestrasyon | `core/pipeline.py` |
| Yapılandırma | `config/settings.py`, `config/banned_words.py`, `config/whitelist.py` |

## Gereksinimler

- **Python 3.10+**
- **FFmpeg** sistemde kurulu ve `PATH` üzerinde erişilebilir olmalı ([indirme](https://ffmpeg.org/download.html))
  - Windows: `winget install ffmpeg` veya binary'yi PATH'e ekleyin
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
- **Vosk Türkçe modeli:** [Vosk modelleri](https://alphacephei.com/vosk/models) sayfasından `vosk-model-tr-*` indirip `model/` içine açın → ayrıntılar için [model/README.md](model/README.md)

## Kurulum

```bash
cd VoxSentinel
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Ek bağımlılıklar:

| Kullanım | Komut |
|----------|-------|
| Mikrofon aracı (`mic_censor.py`) | `pip install -r requirements-mic.txt` |
| Yardımcı scriptler (`scripts/`) | `pip install -r requirements-dev.txt` |

> `openai-whisper` PyTorch ile birlikte gelir; kurulum boyutu büyüktür.
> Yalnızca OpenAI Whisper API kullanacaksanız `WHISPER_MODE=api` ayarlayıp API anahtarı tanımlayın.

## Ortam değişkenleri

| Değişken | Açıklama |
|----------|----------|
| `OPENAI_API_KEY` | Whisper API modu için gerekli. Tanımlı değilse `auto` modda yerel Whisper devreye girer. |
| `VOXSENTINEL_WHISPER_MODE` | `api` \| `local` \| `auto` — `config/settings.py` varsayılanını override eder. |

Anahtarları repoya eklemeyin; `.env` kullanıyorsanız `.gitignore` ile zaten dışlanır.

## Kullanım

### Dosya tabanlı pipeline

```bash
# Temel kullanım (çıktı: giris_sansurlu.wav)
python main.py giris.wav

# Çıkış dosyası belirt
python main.py giris.wav cikis.wav

# Çıkış formatı seç (wav / mp3 / ogg)
python main.py giris.mp3 cikis.mp3 --format mp3

# Yerel Whisper modeli zorla (API anahtarı gerektirmez)
VOXSENTINEL_WHISPER_MODE=local python main.py giris.wav
```

Başlangıçta otomatik durum raporu yazdırılır:

```
  Whisper modu      : AUTO → local
  Yerel model boyutu: small
  Vosk model yolu   : C:\...\model
  Yasaklı kelimeler : 38 adet
  Fonetik eşik      : 0.78
  FFmpeg            : OK
```

### Mikrofon

```bash
# Tam mod: kaydet → pipeline → çal
python mic_censor.py

# Akış modu: ~4s chunk, Vosk+Fonetik, düşük gecikme
python mic_censor.py --stream
```

Windows hızlı mikrofon testi: `CIHAZ_SEC.bat` veya `python scripts\_mic_test_basit.py`

### Yardımcı scriptler

```bash
python scripts\_cihaz_sec.py
python scripts\test_mic.py
```

`scripts/_bootstrap_path.py` proje kökünü `sys.path`'e ekler; içe aktarmalar bu sayede çalışır.

## Yapılandırma

Tüm parametreler `config/settings.py` içinde merkezi olarak yönetilir:

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `WHISPER_MODE` | `"auto"` | `api` / `local` / `auto` |
| `WHISPER_LOCAL_MODEL_SIZE` | `"small"` | `tiny` / `base` / `small` / `medium` / `large` |
| `WHISPER_API_TIMEOUT` | `60` | API çağrısı zaman aşımı (sn) |
| `WHISPER_RETRY_MAX` | `3` | API geçici hatasında yeniden deneme |
| `PHONETIC_SIMILARITY_THRESHOLD` | `0.78` | Jaro-Winkler eşiği (0–1) |
| `PHONETIC_LENGTH_DIFF_MAX` | `3` | Fonetik karşılaştırma maks. uzunluk farkı |
| `BIGRAM_MAX_GAP_SEC` | `0.30` | Bigram birleştirme maks. boşluk (sn) |
| `CENSOR_PADDING_MS` | `60` | Bip kenar dolgusu (ms) |
| `VOTE_MERGE_THRESHOLD_MS` | `80` | Yakın segment birleştirme eşiği (ms) |
| `MAX_RECORDING_DURATION_SEC` | `300` | Mikrofon tam mod maks. kayıt süresi (sn) |

## Yasaklı kelime listesi

`config/banned_words.py` dosyasında üç seviyeli yapı:

| Seviye | Açıklama |
|--------|----------|
| `yuksek` | Ağır cinsel/bedensel küfürler |
| `orta` | Hakaret / aşağılama |
| `dusuk` | Hafif hakaret / alay |

Yalnızca **kök** kelimeler eklenir; türevler, ekler ve fonetik benzerler otomatik yakalanır. Sonuç raporunda seviye dağılımı gösterilir:

```
  Seviye dagılımı: Agir=2  Orta=1  Hafif=0
```

## Dizin yapısı

```
VoxSentinel/
├── main.py              # Dosya → pipeline (CLI)
├── mic_censor.py        # Mikrofon / akış sansürü
├── requirements.txt
├── requirements-mic.txt
├── requirements-dev.txt
├── CIHAZ_SEC.bat
├── core/
│   └── pipeline.py      # Orkestrasyon (3 katman)
├── asr/
│   ├── vosk_engine.py
│   ├── whisper_engine.py
│   ├── whisper_local_engine.py
│   └── phonetic_matcher.py
├── audio/
│   ├── censor_processor.py
│   └── converter.py
├── decision/
│   ├── time_aligner.py
│   └── voting_engine.py
├── config/
│   ├── settings.py
│   ├── banned_words.py
│   └── whitelist.py
├── scripts/             # Teşhis ve deneme araçları
└── model/               # README + sizin indirdiğiniz Vosk modeli (git'e girmez)
```

## Yasal ve etik uyarı

Otomatik sansür **yanlış pozitif/negatif** üretebilir. Bu yazılımı yalnızca yasalara ve kurum/kişi politikalarına uygun şekilde kullanın. Canlı ortamda kullanmadan önce kendi veri kümenizle doğrulama yapın.

## Uzak depo

```
https://github.com/Gokhancagaptay/vox-sentinel.git
```
