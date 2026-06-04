"""
pytest konfigürasyonu — proje kökünü sys.path'e ekler.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fake_vosk_words():
    """Temiz (yasaklı kelime içermeyen) Vosk çıktısı."""
    return [
        {"word": "merhaba", "start": 0.10, "end": 0.50, "conf": 0.90},
        {"word": "dünya",   "start": 1.00, "end": 1.40, "conf": 0.85},
    ]


@pytest.fixture
def fake_whisper_words():
    """Temiz (yasaklı kelime içermeyen) Whisper çıktısı."""
    return [
        {"word": "merhaba", "start": 0.12, "end": 0.52},
        {"word": "dünya",   "start": 1.02, "end": 1.42},
    ]


@pytest.fixture
def banned_whisper_words():
    """'aptal' yasaklı kelimesini içeren Whisper çıktısı."""
    return [
        {"word": "merhaba", "start": 0.12, "end": 0.52},
        {"word": "aptal",   "start": 1.02, "end": 1.42},
    ]


@pytest.fixture
def phonetic_vosk_words():
    """'aptal' içeren Vosk çıktısı (fonetik tespit için)."""
    return [
        {"word": "merhaba", "start": 0.10, "end": 0.50, "conf": 0.90},
        {"word": "aptal",   "start": 1.00, "end": 1.40, "conf": 0.70},
    ]
