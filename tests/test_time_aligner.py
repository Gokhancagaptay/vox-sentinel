"""
Testler: decision/time_aligner.py
- build_anchor_map() DP-LCS doğruluğu
- map_vosk_time_to_whisper() interpolasyon / ekstrapolasyon / anchor yok
- align_phonetic_detections() zaman güncelleme
"""

import pytest

from decision.time_aligner import (
    align_phonetic_detections,
    build_anchor_map,
    map_vosk_time_to_whisper,
)


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


# ── build_anchor_map (DP-LCS) ────────────────────────────────────────────────


class TestBuildAnchorMap:

    def test_bos_listeler(self):
        assert build_anchor_map([], []) == []

    def test_bos_vosk(self):
        whisper = [_w("merhaba", 0.0, 0.5)]
        assert build_anchor_map([], whisper) == []

    def test_bos_whisper(self):
        vosk = [_w("merhaba", 0.0, 0.5)]
        assert build_anchor_map(vosk, []) == []

    def test_tam_eslesme(self):
        vosk = [_w("merhaba", 0.1, 0.5), _w("dunya", 0.6, 1.0)]
        whisper = [_w("merhaba", 0.15, 0.55), _w("dunya", 0.65, 1.05)]
        anchors = build_anchor_map(vosk, whisper)
        assert len(anchors) == 2
        assert anchors[0] == (0.1, 0.15)
        assert anchors[1] == (0.6, 0.65)

    def test_kismi_eslesme(self):
        vosk = [_w("a", 0.0, 0.3), _w("b", 0.4, 0.7), _w("c", 0.8, 1.0)]
        whisper = [_w("x", 0.0, 0.3), _w("b", 0.45, 0.75), _w("y", 0.9, 1.1)]
        anchors = build_anchor_map(vosk, whisper)
        assert len(anchors) == 1
        assert anchors[0][0] == 0.4  # vosk "b"
        assert anchors[0][1] == 0.45  # whisper "b"

    def test_hicbir_eslesme_yok(self):
        vosk = [_w("a", 0.0, 0.5)]
        whisper = [_w("b", 0.0, 0.5)]
        anchors = build_anchor_map(vosk, whisper)
        assert anchors == []

    def test_dp_lcs_greedy_den_daha_uzun(self):
        """DP-LCS greedy'nin kaçırabileceği durumu yakalamalı."""
        vosk = [_w("a", 0.0, 0.1), _w("b", 0.2, 0.3), _w("c", 0.4, 0.5)]
        whisper = [_w("a", 0.0, 0.1), _w("x", 0.15, 0.2), _w("b", 0.25, 0.35), _w("c", 0.45, 0.55)]
        anchors = build_anchor_map(vosk, whisper)
        assert len(anchors) == 3  # a, b, c hepsi eşleşmeli

    def test_anchor_siralamasi_kronolojik(self):
        vosk = [_w("x", 0.0, 0.1), _w("y", 0.5, 0.6)]
        whisper = [_w("x", 0.05, 0.15), _w("y", 0.55, 0.65)]
        anchors = build_anchor_map(vosk, whisper)
        vosk_times = [a[0] for a in anchors]
        assert vosk_times == sorted(vosk_times)

    def test_tekrar_eden_kelimeler(self):
        """Aynı kelime birden fazla geçiyorsa LCS doğru sayıyı bulmalı."""
        vosk = [_w("a", 0.0, 0.1), _w("a", 0.5, 0.6)]
        whisper = [_w("a", 0.05, 0.15), _w("a", 0.55, 0.65)]
        anchors = build_anchor_map(vosk, whisper)
        assert len(anchors) == 2


# ── map_vosk_time_to_whisper ─────────────────────────────────────────────────


class TestMapVoskTimeToWhisper:

    def test_anchor_yok_vosk_zamani_doner(self):
        result = map_vosk_time_to_whisper(1.5, [])
        assert result >= 0.0

    def test_interpolasyon_iki_anchor_arasi(self):
        # anchors = [(v1=0.0, w1=0.1), (v2=1.0, w2=1.2)]
        # vosk=0.5 → ratio=0.5 → w1 + 0.5*(w2-w1) = 0.1 + 0.5*1.1 = 0.65
        anchors = [(0.0, 0.1), (1.0, 1.2)]
        result = map_vosk_time_to_whisper(0.5, anchors)
        assert pytest.approx(result, abs=0.01) == 0.65

    def test_interpolasyon_linearity(self):
        anchors = [(0.0, 0.0), (2.0, 4.0)]  # 2x hız farkı
        assert pytest.approx(map_vosk_time_to_whisper(1.0, anchors), abs=0.001) == 2.0

    def test_anchor_sol_ekstrapolasyon(self):
        anchors = [(0.0, 0.1), (1.0, 1.1)]  # 0.1s sabit offset
        result = map_vosk_time_to_whisper(2.0, anchors)
        assert result > 1.0  # zamandan büyük olmalı

    def test_negatif_zaman_kliplanir(self):
        anchors = [(1.0, 0.5)]
        result = map_vosk_time_to_whisper(0.0, anchors)
        assert result >= 0.0

    def test_tek_anchor_tam_ustunde(self):
        anchors = [(1.0, 1.5)]
        result = map_vosk_time_to_whisper(1.0, anchors)
        assert pytest.approx(result, abs=0.001) == 1.5


# ── align_phonetic_detections ────────────────────────────────────────────────


class TestAlignPhoneticDetections:

    def test_bos_liste(self):
        result = align_phonetic_detections([], [(0.0, 0.1)])
        assert result == []

    def test_zaman_guncellenir(self):
        anchors = [(0.0, 0.1), (2.0, 2.2)]
        detections = [
            {
                "word": "pis",
                "start": 1.0,
                "end": 1.3,
                "matched_banned": "piç",
                "source": "phonetic",
                "similarity_score": 0.89,
            }
        ]
        result = align_phonetic_detections(detections, anchors)
        assert len(result) == 1
        assert result[0]["start"] != 1.0  # güncellendi
        assert "vosk_start_original" in result[0]

    def test_orijinal_zaman_korunur(self):
        anchors = [(0.0, 0.5)]
        det = [
            {
                "word": "t",
                "start": 0.5,
                "end": 0.8,
                "matched_banned": "x",
                "source": "phonetic",
                "similarity_score": 0.8,
            }
        ]
        result = align_phonetic_detections(det, anchors)
        assert result[0]["vosk_start_original"] == 0.5
