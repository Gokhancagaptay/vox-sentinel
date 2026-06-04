"""
Testler: asr/phonetic_matcher.py
- find_phonetic_match() doğru eşleşme / yanlış pozitif / kısa kelime
- scan_for_phonetic_matches() toplu tarama
- _BANNED_NORMALIZED önbelleki
"""

import pytest

from asr.phonetic_matcher import (
    _BANNED_NORMALIZED,
    YASAKLI_KELIMELER,
    find_phonetic_match,
    scan_for_phonetic_matches,
)

# ── find_phonetic_match ──────────────────────────────────────────────────────


class TestFindPhoneticMatch:

    def test_tam_eslesme(self):
        matched, score = find_phonetic_match("piç")
        assert matched == "piç"
        assert score == pytest.approx(1.0, abs=0.01)

    def test_yakin_ses_yakalama(self):
        """'pis' → 'piç' fonetik olarak yakalanmalı."""
        matched, score = find_phonetic_match("pis")
        assert matched == "piç"
        assert score > 0.78

    def test_uzak_kelime_gecmeli(self):
        """'merhaba' hiçbir yasaklı kelimeyle eşleşmemeli."""
        matched, score = find_phonetic_match("merhaba")
        assert matched is None
        assert score == 0.0

    def test_cok_kisa_kelime_atlanir(self):
        """2 karakter → fonetik karşılaştırma yapılmamalı."""
        matched, score = find_phonetic_match("bu")
        assert matched is None

    def test_bos_kelime(self):
        matched, score = find_phonetic_match("")
        assert matched is None

    def test_buyuk_kucuk_harf_farketmez(self):
        matched_kucuk, _ = find_phonetic_match("aptal")
        matched_buyuk, _ = find_phonetic_match("APTAL")
        assert matched_kucuk == matched_buyuk

    def test_bosluk_temizleme(self):
        matched, _ = find_phonetic_match("  aptal  ")
        assert matched == "aptal"

    def test_esik_ustu_yakalar(self):
        """'siki' → 'sik' alt-dize değil ama fonetik çok yakın."""
        matched, score = find_phonetic_match("siki")
        assert matched is not None

    def test_hicbir_eslesme_yok_bos_liste(self):
        matched, score = find_phonetic_match("aptal", banned_list=[])
        assert matched is None
        assert score == 0.0


# ── scan_for_phonetic_matches ────────────────────────────────────────────────


class TestScanForPhoneticMatches:

    def test_bos_liste(self):
        result = scan_for_phonetic_matches([])
        assert result == []

    def test_temiz_kelimeler(self):
        words = [
            {"word": "merhaba", "start": 0.0, "end": 0.5},
            {"word": "nasılsın", "start": 0.6, "end": 1.2},
        ]
        result = scan_for_phonetic_matches(words)
        assert result == []

    def test_kufurlu_kelime_yakalama(self):
        words = [
            {"word": "merhaba", "start": 0.0, "end": 0.5},
            {"word": "pis", "start": 1.0, "end": 1.3},
        ]
        result = scan_for_phonetic_matches(words)
        assert len(result) == 1
        assert result[0]["source"] == "phonetic"
        assert result[0]["matched_banned"] is not None
        assert "start" in result[0]
        assert "end" in result[0]

    def test_cikti_yapisi(self):
        words = [{"word": "aptal", "start": 2.0, "end": 2.4}]
        result = scan_for_phonetic_matches(words)
        assert len(result) == 1
        r = result[0]
        assert set(r.keys()) >= {
            "word",
            "start",
            "end",
            "matched_banned",
            "similarity_score",
            "source",
        }
        assert r["source"] == "phonetic"
        assert 0.0 <= r["similarity_score"] <= 1.0


# ── Önbellek ────────────────────────────────────────────────────────────────


class TestBannedNormalizedCache:

    def test_onbellek_uzunlugu(self):
        assert len(_BANNED_NORMALIZED) == len(YASAKLI_KELIMELER)

    def test_onbellek_kucuk_harf(self):
        for norm in _BANNED_NORMALIZED:
            assert norm == norm.lower()
