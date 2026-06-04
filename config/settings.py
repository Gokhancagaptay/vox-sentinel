"""
Uygulama genelinde merkezi yapılandırma ayarları.
Tüm sabit değerler buradan yönetilir; kodun içine gömülmez.

Ortam değişkeni override:
    VOXSENTINEL_WHISPER_MODE=api|local|auto  (varsayılan: auto)
"""

import os
from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).parent.parent

# ─── Vosk Ayarları ───────────────────────────────────────────────
VOSK_MODEL_PATH = str(BASE_DIR / "model")
VOSK_SAMPLE_RATE = 16000
VOSK_CHUNK_SIZE = 4000  # Her turda okunacak frame sayısı

# Vosk'un gerektirdiği WAV formatı (converter.py tarafından kullanılır)
VOSK_REQUIRED_CHANNELS = 1  # Mono
VOSK_REQUIRED_SAMPLE_RATE = VOSK_SAMPLE_RATE  # 16 kHz
VOSK_REQUIRED_SAMPLE_WIDTH = 2  # 16-bit (2 byte)

# ─── Whisper API Ayarları ─────────────────────────────────────────
WHISPER_MODEL = "whisper-1"
WHISPER_LANGUAGE = "tr"  # Türkçe zorla; None yapılırsa otomatik algılar
WHISPER_RESPONSE_FORMAT = "verbose_json"
WHISPER_API_TIMEOUT = 60  # API çağrısı için zaman aşımı (saniye)
WHISPER_RETRY_MAX = 3  # Geçici API hatasında maksimum yeniden deneme

# ─── Yerel Whisper Ayarları ───────────────────────────────────────
# Model boyutu / hız / doğruluk dengesi (CPU için önerilen: "base" veya "small")
#   tiny   ~75MB  | çok hızlı  | düşük doğruluk
#   base   ~150MB | hızlı      | iyi denge
#   small  ~490MB | orta hız   | daha iyi       ← varsayılan (Türkçe küfür için önerilir)
#   medium ~1.5GB | yavaş      | yüksek doğruluk
#   large  ~3GB   | çok yavaş  | en yüksek (GPU önerilir)
WHISPER_LOCAL_MODEL_SIZE = "small"

# Yerel Whisper model dosyalarının indirileceği/okunacağı dizin
WHISPER_LOCAL_MODEL_DIR = str(BASE_DIR / "whisper_models")

# Yerel Whisper backend seçimi:
#   "faster-whisper" → CTranslate2 tabanlı, 4× hızlı, az bellek (önerilen)
#   "openai-whisper" → Orijinal PyTorch tabanlı
# faster-whisper kurulu değilse otomatik olarak openai-whisper'a düşer.
WHISPER_LOCAL_BACKEND = "faster-whisper"

# Çalışma modu — ortam değişkeni ile override edilebilir:
# "api"   → OpenAI Whisper API (OPENAI_API_KEY gerekli, ücretli)
# "local" → Yerel model (ücretsiz, çevrimdışı)
# "auto"  → API anahtarı varsa API, yoksa yerel model
_mode_raw = os.environ.get("VOXSENTINEL_WHISPER_MODE", "auto").lower()
WHISPER_MODE = _mode_raw if _mode_raw in ("api", "local", "auto") else "auto"

# ─── Fonetik Eşleştirici Ayarları ────────────────────────────────
# Jaro-Winkler benzerlik eşiği: 0.0 (hiç benzer değil) → 1.0 (tam eşleşme)
# 0.78: "hiçbir"↔"piç" (0.71) geçer ama "pis"↔"piç" (0.89) ve "siki"↔"sik" (0.94) yakalar
# Not: Eşiği çok düşürmek yanlış pozitiflere yol açar; 0.78 Türkçe için iyi denge noktası
PHONETIC_SIMILARITY_THRESHOLD = 0.78
# Kelime uzunluğu farkı bu değeri aşarsa fonetik karşılaştırma yapılmaz.
# Hem phonetic_matcher.py hem voting_engine.py bu değeri kullanır (önceden her dosyada farklıydı).
PHONETIC_LENGTH_DIFF_MAX = 3
# Jaro-Winkler hesaplamalarını önbelleğe alan lru_cache boyutu (38 banned × ~108 tekil kelime)
PHONETIC_CACHE_MAXSIZE = 4096

# ─── Bip Sesi Ayarları ───────────────────────────────────────────
BEEP_FREQUENCY_HZ = 1000  # Klasik TV sansür bip frekansı
BEEP_GAIN_DB = -5  # Bip sesinin şiddet ayarı (dB)

# Sansür kesiminin kelimeden önce/sonra ne kadar uzayacağı (ms)
# Telaffuz başlangıç/bitiş kaymalarını telafi eder
CENSOR_PADDING_MS = 60
# Padding sonrası süresi bu değerin altına düşen segmentler atlanır
MIN_SEGMENT_DURATION_MS = 10

# ─── Oylama Motoru Ayarları ──────────────────────────────────────
# Bu kadar ms içindeki tespitler tek bir sansür segmentinde birleştirilir.
# Düşük değer → ayrı kelimeler ayrı bip alır (tercih edilen)
# Yüksek değer → yan yana küfürler tek bip'e birleşir
VOTE_MERGE_THRESHOLD_MS = 80
# Bigram kontrolünde iki ardışık kelime arasındaki maksimum boşluk (saniye)
BIGRAM_MAX_GAP_SEC = 0.30

# ─── Uzun Ses Bölümleme (Chunking) ───────────────────────────────
# Yerel Whisper modelinde uzun dosyalar için otomatik parçalı işleme
WHISPER_CHUNK_THRESHOLD_SEC = 60  # Bu süreyi aşan dosyalar parçalanır
WHISPER_CHUNK_DURATION_SEC = 30  # Her parçanın süresi (saniye)
WHISPER_CHUNK_OVERLAP_SEC = 5  # Parça başı/sonu örtüşme (kelime kırılmasını önler)

# ─── Mikrofon / Kayıt Ayarları ───────────────────────────────────
# tam_mod() kayıt güvenlik limiti; bu süreyi aşan kayıt otomatik durdurulur
MAX_RECORDING_DURATION_SEC = 300
