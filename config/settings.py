"""
Uygulama genelinde merkezi yapılandırma ayarları.
Tüm sabit değerler buradan yönetilir; kodun içine gömülmez.
"""

from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).parent.parent

# ─── Vosk Ayarları ───────────────────────────────────────────────
VOSK_MODEL_PATH = str(BASE_DIR / "model")
VOSK_SAMPLE_RATE = 16000
VOSK_CHUNK_SIZE = 4000  # Her turda okunacak frame sayısı

# ─── Whisper API Ayarları ─────────────────────────────────────────
WHISPER_MODEL = "whisper-1"
WHISPER_LANGUAGE = "tr"          # Türkçe zorla; None yapılırsa otomatik algılar
WHISPER_RESPONSE_FORMAT = "verbose_json"

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

# Çevririm içi çalışma modu:
# "api"   → OpenAI Whisper API (OPENAI_API_KEY gerekli, ücretli)
# "local" → Yerel model (ücretsiz, çevrimdışı)
# "auto"  → API anahtarı varsa API, yoksa yerel model
WHISPER_MODE = "auto"

# ─── Fonetik Eşleştirici Ayarları ────────────────────────────────
# Jaro-Winkler benzerlik eşiği: 0.0 (hiç benzer değil) → 1.0 (tam eşleşme)
# 0.78: "hiçbir"↔"piç" (0.71) geçer ama "pis"↔"piç" (0.89) ve "siki"↔"sik" (0.94) yakalar
# Not: Eşiği çok düşürmek yanlış pozitiflere yol açar; 0.78 Türkçe için iyi denge noktası
PHONETIC_SIMILARITY_THRESHOLD = 0.78

# ─── Bip Sesi Ayarları ───────────────────────────────────────────
BEEP_FREQUENCY_HZ = 1000   # Klasik TV sansür bip frekansı
BEEP_GAIN_DB = -5           # Bip sesinin şiddet ayarı (dB)

# Sansür kesiminin kelimeden önce/sonra ne kadar uzayacağı (ms)
# Telaffuz başlangıç/bitiş kaymalarını telafi eder
CENSOR_PADDING_MS = 60

# ─── Oylama Motoru Ayarları ──────────────────────────────────────
# Bu kadar ms içindeki tespitler tek bir sansür segmentinde birleştirilir.
# Düşük değer → ayrı kelimeler ayrı bip alır (tercih edilen)
# Yüksek değer → yan yana küfürler tek bip'e birleşir
VOTE_MERGE_THRESHOLD_MS = 80
