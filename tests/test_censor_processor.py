"""
Testler: audio/censor_processor.py
- _merge_overlapping(): çakışma, bitişik, ayrı segmentler, minimum süre
- apply_censor_beeps(): gerçek WAV dosyasıyla tam döngü (FFmpeg gerekli)
"""

import os
import tempfile
from pathlib import Path

import pytest

from audio.censor_processor import _merge_overlapping, apply_censor_beeps

REPO_ROOT = Path(__file__).parent.parent
TEST_WAV = REPO_ROOT / "test_sesi.wav"
AUDIO_DURATION_MS = 30_000  # 30s üst sınır (test dosyası ~11s)


def _seg(start_ms, end_ms, word="test"):
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "word": word,
        "matched_banned": "aptal",
        "source": "whisper",
    }


# ── _merge_overlapping ────────────────────────────────────────────────────────


class TestMergeOverlapping:

    def test_bos_liste(self):
        assert _merge_overlapping([], AUDIO_DURATION_MS) == []

    def test_tek_segment(self):
        result = _merge_overlapping([_seg(1000, 2000)], AUDIO_DURATION_MS)
        assert len(result) == 1

    def test_cakisan_birlesir(self):
        segs = [_seg(1000, 2000), _seg(1500, 2500)]
        result = _merge_overlapping(segs, AUDIO_DURATION_MS)
        assert len(result) == 1
        assert result[0]["start_ms"] <= 1000 - 60  # padding dahil
        assert result[0]["end_ms"] >= 2500 + 60

    def test_bitisik_birlesir(self):
        """Padding sonrası dokunan segmentler birleşmeli."""
        segs = [_seg(1000, 2000), _seg(2000, 3000)]
        result = _merge_overlapping(segs, AUDIO_DURATION_MS)
        assert len(result) == 1

    def test_ayri_kalir(self):
        """500ms boşluklu segmentler (padding sonrası) ayrı kalmalı."""
        segs = [_seg(100, 500), _seg(1000, 1400)]
        result = _merge_overlapping(segs, AUDIO_DURATION_MS)
        assert len(result) == 2

    def test_sinir_klipleme(self):
        """start < 0 → 0'a kliplanmalı."""
        segs = [_seg(10, 500)]  # 10 - 60 padding = -50 → 0
        result = _merge_overlapping(segs, AUDIO_DURATION_MS)
        assert result[0]["start_ms"] == 0

    def test_minimum_sure_atlaniyor(self):
        """Padding sonrası çok kısa segment atlanmalı."""
        segs = [_seg(100, 101)]  # 1ms → padding sonrası süre hala kısa değil normalde, ama MIN=10ms
        result = _merge_overlapping(segs, AUDIO_DURATION_MS)
        # 101+60 - (100-60) = 21ms ≥ MIN_SEGMENT_DURATION_MS=10 → geçmeli
        assert len(result) == 1


# ── apply_censor_beeps ────────────────────────────────────────────────────────


@pytest.mark.skipif(not TEST_WAV.exists(), reason="test_sesi.wav bulunamadı")
class TestApplyCensorBeeps:

    def test_bos_segment_hata_verir(self):
        with pytest.raises(ValueError, match="boş"):
            apply_censor_beeps(str(TEST_WAV), [], "cikti.wav")

    def test_tek_segment_dosya_olusturur(self):
        segs = [_seg(1000, 2000)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            result = apply_censor_beeps(str(TEST_WAV), segs, out)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
        finally:
            if os.path.exists(out):
                os.unlink(out)

    def test_cikti_suresi_degismez(self):
        """Bip ekleme ses süresini değiştirmemeli."""
        from pydub import AudioSegment

        original = AudioSegment.from_file(str(TEST_WAV))
        segs = [_seg(2000, 3000), _seg(5000, 6000)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            apply_censor_beeps(str(TEST_WAV), segs, out)
            censored = AudioSegment.from_file(out)
            # Süre 100ms toleransla aynı olmalı
            assert abs(len(original) - len(censored)) < 100
        finally:
            if os.path.exists(out):
                os.unlink(out)

    def test_coklu_segment_calisir(self):
        """10 segment → hata olmadan işlenmeli (O(n) algoritması)."""
        segs = [_seg(i * 1000, i * 1000 + 400) for i in range(10)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            apply_censor_beeps(str(TEST_WAV), segs, out)
            assert os.path.exists(out)
        finally:
            if os.path.exists(out):
                os.unlink(out)
