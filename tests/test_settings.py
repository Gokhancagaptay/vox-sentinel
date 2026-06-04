"""
Testler: config/settings.py
- Tüm gerekli değişkenler tanımlı
- Değerler makul aralıkta
- Env var override çalışıyor
"""

import importlib
import os


def _reload_settings(env_override=None):
    """settings'i temiz env ile yeniden yükle."""
    old = os.environ.copy()
    if env_override:
        os.environ.update(env_override)
    try:
        import config.settings as s

        importlib.reload(s)
        return s
    finally:
        os.environ.clear()
        os.environ.update(old)


class TestSettingsDefaults:

    def setup_method(self):
        import config.settings as s

        self.s = s

    def test_whisper_mode_gecerli(self):
        assert self.s.WHISPER_MODE in ("api", "local", "auto")

    def test_api_timeout_pozitif(self):
        assert self.s.WHISPER_API_TIMEOUT > 0

    def test_retry_max_pozitif(self):
        assert self.s.WHISPER_RETRY_MAX >= 1

    def test_phonetic_esik_aralikta(self):
        assert 0.0 < self.s.PHONETIC_SIMILARITY_THRESHOLD < 1.0

    def test_phonetic_uzunluk_fark_pozitif(self):
        assert self.s.PHONETIC_LENGTH_DIFF_MAX > 0

    def test_bigram_gap_pozitif(self):
        assert self.s.BIGRAM_MAX_GAP_SEC > 0

    def test_censor_padding_pozitif(self):
        assert self.s.CENSOR_PADDING_MS > 0

    def test_vote_merge_pozitif(self):
        assert self.s.VOTE_MERGE_THRESHOLD_MS > 0

    def test_min_segment_pozitif(self):
        assert self.s.MIN_SEGMENT_DURATION_MS > 0

    def test_max_recording_makul(self):
        assert 30 <= self.s.MAX_RECORDING_DURATION_SEC <= 3600

    def test_beep_frekans_duyulabilir(self):
        assert 200 <= self.s.BEEP_FREQUENCY_HZ <= 8000

    def test_vosk_chunk_pozitif(self):
        assert self.s.VOSK_CHUNK_SIZE > 0


class TestEnvOverride:

    def test_gecerli_mod_override(self):
        s = _reload_settings({"VOXSENTINEL_WHISPER_MODE": "local"})
        assert s.WHISPER_MODE == "local"

    def test_gecersiz_mod_auto_donusu(self):
        s = _reload_settings({"VOXSENTINEL_WHISPER_MODE": "gecersiz_deger"})
        assert s.WHISPER_MODE == "auto"

    def test_api_mod_override(self):
        s = _reload_settings({"VOXSENTINEL_WHISPER_MODE": "api"})
        assert s.WHISPER_MODE == "api"

    def test_buyuk_harf_normalize(self):
        s = _reload_settings({"VOXSENTINEL_WHISPER_MODE": "LOCAL"})
        assert s.WHISPER_MODE == "local"
