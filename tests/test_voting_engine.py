"""
Testler: decision/voting_engine.py
- find_whisper_banned_words(): tek kelime, bigram, dedup
- vote_and_merge(): birleştirme, çakışma, sıralama
- _normalize(), _fuzzy_threshold_for()
"""

from decision.voting_engine import (
    _fuzzy_threshold_for,
    _normalize,
    find_whisper_banned_words,
    vote_and_merge,
)


def _word(w, start, end):
    return {"word": w, "start": start, "end": end}


# ── _normalize ────────────────────────────────────────────────────────────────


class TestNormalize:

    def test_kucuk_harf(self):
        assert _normalize("APTAL") == "aptal"

    def test_bosluk_temizleme(self):
        assert _normalize("  sik  ") == "sik"

    def test_turkce_karakter(self):
        result = _normalize("ŞEREFSIZ")
        assert result == "şerefsiz" or result == "şerefsız"  # NFC


# ── _fuzzy_threshold_for ─────────────────────────────────────────────────────


class TestFuzzyThreshold:

    def test_3_karakter_alti_kapali(self):
        t = _fuzzy_threshold_for("pi")
        assert t > 1.0  # etkin olarak kapalı

    def test_3_karakter_kapali(self):
        t = _fuzzy_threshold_for("piç")
        assert t > 1.0

    def test_4_5_karakter_yuksek(self):
        t = _fuzzy_threshold_for("aptal")
        assert 0.85 <= t <= 0.95

    def test_6_uzun_normal(self):
        t = _fuzzy_threshold_for("şerefsiz")
        assert 0.75 <= t <= 0.85


# ── find_whisper_banned_words ─────────────────────────────────────────────────


class TestFindWhisperBannedWords:

    def test_bos_liste(self):
        assert find_whisper_banned_words([]) == []

    def test_tam_eslesme_yakalama(self):
        words = [_word("aptal", 1.0, 1.4)]
        result = find_whisper_banned_words(words)
        assert len(result) == 1
        assert result[0]["matched_banned"] == "aptal"
        assert result[0]["source"] == "whisper"

    def test_alt_dize_yakalama(self):
        """'aptallık' içinde 'aptal' alt-dize eşleşmesi."""
        words = [_word("aptallık", 0.5, 1.0)]
        result = find_whisper_banned_words(words)
        assert len(result) >= 1

    def test_temiz_kelimeler_gecmeli(self):
        words = [_word("merhaba", 0.0, 0.5), _word("güzel", 0.6, 1.0)]
        result = find_whisper_banned_words(words)
        assert result == []

    def test_dedup_ayni_start_end(self):
        """Aynı start/end'e sahip iki tespit → sadece bir kez raporlanmalı."""
        words = [_word("sik", 1.0, 1.3), _word("sik", 1.0, 1.3)]
        result = find_whisper_banned_words(words)
        assert len(result) == 1

    def test_cikti_yapisi(self):
        words = [_word("aptal", 0.0, 0.4)]
        result = find_whisper_banned_words(words)
        assert len(result) > 0
        r = result[0]
        assert "word" in r
        assert "start" in r
        assert "end" in r
        assert "matched_banned" in r
        assert "source" in r

    def test_buyuk_kucuk_harf_farketmez(self):
        r1 = find_whisper_banned_words([_word("Aptal", 0.0, 0.4)])
        r2 = find_whisper_banned_words([_word("aptal", 0.0, 0.4)])
        assert (len(r1) > 0) == (len(r2) > 0)


# ── vote_and_merge ────────────────────────────────────────────────────────────


class TestVoteAndMerge:

    def _det(self, start_s, end_s, word="test", matched="aptal", source="whisper"):
        return {
            "start": start_s,
            "end": end_s,
            "word": word,
            "matched_banned": matched,
            "source": source,
        }

    def test_bos_girdi(self):
        assert vote_and_merge([], []) == []

    def test_sadece_whisper(self):
        result = vote_and_merge([self._det(1.0, 1.5)], [])
        assert len(result) == 1
        assert result[0]["start_ms"] == 1000
        assert result[0]["end_ms"] == 1500

    def test_sadece_fonetik(self):
        result = vote_and_merge([], [self._det(2.0, 2.5, source="phonetic")])
        assert len(result) == 1

    def test_cakisan_segmentler_birlesir(self):
        w = [self._det(1.0, 1.5), self._det(1.4, 2.0)]
        result = vote_and_merge(w, [])
        assert len(result) == 1
        assert result[0]["start_ms"] == 1000
        assert result[0]["end_ms"] == 2000

    def test_yakin_segmentler_birlesir(self):
        """80ms eşiği içindeki segmentler birleşmeli."""
        w = [self._det(1.0, 1.3), self._det(1.35, 1.6)]
        result = vote_and_merge(w, [], merge_threshold_ms=80)
        assert len(result) == 1

    def test_uzak_segmentler_ayri_kalir(self):
        """500ms uzaktaki segmentler ayrı kalmalı."""
        w = [self._det(0.0, 0.3), self._det(1.0, 1.3)]
        result = vote_and_merge(w, [], merge_threshold_ms=80)
        assert len(result) == 2

    def test_zamana_gore_siralanir(self):
        w = [self._det(2.0, 2.5), self._det(0.5, 1.0)]
        result = vote_and_merge(w, [])
        assert result[0]["start_ms"] < result[1]["start_ms"]

    def test_kaynak_birlestirme(self):
        """Aynı aralıkta hem whisper hem fonetik → kaynak birleşmeli."""
        w = [self._det(1.0, 1.5, source="whisper")]
        p = [self._det(1.1, 1.4, source="phonetic")]
        result = vote_and_merge(w, p)
        assert len(result) == 1
        assert "whisper" in result[0]["source"]
        assert "phonetic" in result[0]["source"]

    def test_ms_donusumu_dogruluk(self):
        result = vote_and_merge([self._det(1.234, 2.567)], [])
        assert result[0]["start_ms"] == 1234
        assert result[0]["end_ms"] == 2567
