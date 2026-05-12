"""
Katman 2 — Zaman Hizalama Motoru

Vosk ve Whisper farklı zaman eksenlerinde çalışabilir: aynı ses dosyasını
analiz etseler de VAD (Voice Activity Detection) veya model iç tamponları
nedeniyle aynı kelime için farklı timestamp döndürebilirler.

Bu modülün görevi:
1. Her iki motorda da aynı şekilde geçen kelimeleri "anchor" olarak bulmak.
2. Bu anchor noktaları arasında doğrusal interpolasyon yaparak
   Vosk zaman eksenini Whisper zaman eksenine hizalamak.
3. Fonetik eşleşme tespitlerinin (Vosk zamanında) gerçek ses zamanını
   bulmak için bu haritayı kullanmak.

Algoritma: Greedy LCS (Longest Common Subsequence) tabanlı anchor bulma.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_anchor_map(
    vosk_words: list[dict],
    whisper_words: list[dict],
) -> list[tuple[float, float]]:
    """
    İki kelime listesi arasında zaman hizalama referans noktaları oluşturur.

    Her iki listede de aynı kelimeyi içeren çiftler anchor olarak seçilir.
    Greedy yaklaşım: Vosk listesinde ilerlerken Whisper listesinde ilk
    metin eşleşmesini arar; bu, ses dosyasının doğrusal yapısını korur.

    Args:
        vosk_words    : Vosk'tan gelen kelime listesi.
        whisper_words : Whisper'dan gelen kelime listesi.

    Returns:
        [(vosk_start, whisper_start)] anchor zaman çiftleri listesi.
        Liste boşsa Vosk zamanları doğrudan kullanılır.
    """
    anchors: list[tuple[float, float]] = []
    whisper_search_idx = 0  # Whisper tarafında arama başlangıcı

    for vosk_word in vosk_words:
        vosk_text = vosk_word["word"].lower().strip()

        # Whisper listesinde bu kelimeyi ara (ileri yönlü)
        for w_idx in range(whisper_search_idx, len(whisper_words)):
            whisper_text = whisper_words[w_idx]["word"].lower().strip()

            if vosk_text == whisper_text:
                anchors.append((
                    vosk_word["start"],
                    whisper_words[w_idx]["start"],
                ))
                # Bir sonraki aramayı bu noktanın ötesinden başlat
                whisper_search_idx = w_idx + 1
                break

    logger.debug(
        "[HİZALAMA] %d Vosk kelimesi, %d Whisper kelimesi → %d anchor bulundu.",
        len(vosk_words), len(whisper_words), len(anchors),
    )
    return anchors


def map_vosk_time_to_whisper(
    vosk_time: float,
    anchors: list[tuple[float, float]],
) -> float:
    """
    Verilen Vosk zaman damgasını Whisper zaman eksenine dönüştürür.

    Yöntem:
    - Zaman noktasının solunda ve sağında birer anchor bulunursa
      doğrusal interpolasyon uygulanır.
    - Yalnızca sol anchor varsa (dosya sonu bölgesi) lineer ekstrapolasyon.
    - Yalnızca sağ anchor varsa (dosya başı bölgesi) lineer ekstrapolasyon.
    - Hiç anchor yoksa Vosk zamanı olduğu gibi döndürülür.

    Args:
        vosk_time : Hizalanacak Vosk timestamp (saniye).
        anchors   : build_anchor_map() çıktısı.

    Returns:
        Tahmini Whisper timestamp (saniye).
    """
    if not anchors:
        # Anchor yoksa zaman kayması bilinmiyor; Vosk zamanını kullan
        return vosk_time

    before = [(v, w) for v, w in anchors if v <= vosk_time]
    after  = [(v, w) for v, w in anchors if v >  vosk_time]

    if before and after:
        # ─── Doğrusal interpolasyon ───────────────────────────────
        v1, w1 = before[-1]
        v2, w2 = after[0]
        ratio = (vosk_time - v1) / (v2 - v1) if v2 != v1 else 0.0
        return w1 + ratio * (w2 - w1)

    if before:
        # ─── Sol ekstrapolasyon (dosya sonu) ──────────────────────
        v1, w1 = before[-1]
        if len(before) >= 2:
            v0, w0 = before[-2]
            # İki anchor arasındaki drift oranı (Whisper/Vosk hız farkı)
            drift = (w1 - w0) / (v1 - v0) if v1 != v0 else 1.0
            return w1 + (vosk_time - v1) * drift
        # Tek anchor: sabit kayma varsay (1:1 oranı)
        return w1 + (vosk_time - v1)

    # ─── Sağ ekstrapolasyon (dosya başı) ──────────────────────────
    v1, w1 = after[0]
    return w1 - (v1 - vosk_time)


def align_phonetic_detections(
    phonetic_detections: list[dict],
    anchors: list[tuple[float, float]],
) -> list[dict]:
    """
    Fonetik eşleşme tespitlerinin Vosk zaman damgalarını
    Whisper zaman eksenine hizalar.

    Args:
        phonetic_detections : phonetic_matcher'dan gelen tespit listesi.
        anchors             : build_anchor_map() çıktısı.

    Returns:
        Güncellenmiş start/end ile yeni tespit listesi.
    """
    aligned: list[dict] = []

    for detection in phonetic_detections:
        aligned_start = map_vosk_time_to_whisper(detection["start"], anchors)
        aligned_end   = map_vosk_time_to_whisper(detection["end"],   anchors)

        aligned.append({
            **detection,
            "start":              aligned_start,
            "end":                aligned_end,
            "vosk_start_original": detection["start"],  # Hata ayıklama için koru
            "vosk_end_original":   detection["end"],
        })

    return aligned
