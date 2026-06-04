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

Algoritma: Dinamik Programlama LCS (Longest Common Subsequence) tabanlı anchor bulma.
Greedy LCS'nin aksine DP-LCS en uzun ortak alt diziyi kesin olarak bulur.
"""

import logging

logger = logging.getLogger(__name__)


def build_anchor_map(
    vosk_words: list[dict],
    whisper_words: list[dict],
) -> list[tuple[float, float]]:
    """
    İki kelime listesi arasında zaman hizalama referans noktaları oluşturur.

    Dinamik programlama ile gerçek LCS (En Uzun Ortak Alt Dizi) bulunur.
    Greedy yaklaşımın aksine, kelime tekrarları veya sıra değişikliklerinde
    daha fazla anchor noktası yakalanır.

    Args:
        vosk_words    : Vosk'tan gelen kelime listesi.
        whisper_words : Whisper'dan gelen kelime listesi.

    Returns:
        [(vosk_start, whisper_start)] anchor zaman çiftleri listesi.
        Sıralıdır (vosk_start artan). Liste boşsa Vosk zamanları doğrudan kullanılır.
    """
    n = len(vosk_words)
    m = len(whisper_words)

    if n == 0 or m == 0:
        return []

    vosk_texts = [w["word"].lower().strip() for w in vosk_words]
    whisper_texts = [w["word"].lower().strip() for w in whisper_words]

    # DP tablosu: dp[i][j] = vosk_texts[:i] ve whisper_texts[:j] LCS uzunluğu
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if vosk_texts[i - 1] == whisper_texts[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Geri izleme ile eşleşen indisleri topla
    anchors: list[tuple[float, float]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if vosk_texts[i - 1] == whisper_texts[j - 1]:
            anchors.append(
                (
                    vosk_words[i - 1]["start"],
                    whisper_words[j - 1]["start"],
                )
            )
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    anchors.reverse()  # Geri izleme tersten gelir; kronolojik sıraya çevir

    logger.debug(
        "[HİZALAMA] %d Vosk kelimesi, %d Whisper kelimesi → %d anchor bulundu (DP-LCS).",
        n,
        m,
        len(anchors),
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
        logger.debug("[HİZALAMA] Anchor yok; Vosk zamanı doğrudan kullanılıyor: %.3f", vosk_time)
        return max(0.0, vosk_time)

    before = [(v, w) for v, w in anchors if v <= vosk_time]
    after = [(v, w) for v, w in anchors if v > vosk_time]

    if before and after:
        # ─── Doğrusal interpolasyon ───────────────────────────────
        v1, w1 = before[-1]
        v2, w2 = after[0]
        if abs(v2 - v1) < 1e-6:
            result = w1
        else:
            ratio = (vosk_time - v1) / (v2 - v1)
            result = w1 + ratio * (w2 - w1)
        return max(0.0, result)

    if before:
        # ─── Sol ekstrapolasyon (dosya sonu) ──────────────────────
        v1, w1 = before[-1]
        if len(before) >= 2:
            v0, w0 = before[-2]
            drift = (w1 - w0) / (v1 - v0) if abs(v1 - v0) >= 1e-6 else 1.0
            result = w1 + (vosk_time - v1) * drift
        else:
            result = w1 + (vosk_time - v1)
        return max(0.0, result)

    # ─── Sağ ekstrapolasyon (dosya başı) ──────────────────────────
    v1, w1 = after[0]
    return max(0.0, w1 - (v1 - vosk_time))


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
        aligned_end = map_vosk_time_to_whisper(detection["end"], anchors)

        aligned.append(
            {
                **detection,
                "start": aligned_start,
                "end": aligned_end,
                "vosk_start_original": detection["start"],  # Hata ayıklama için koru
                "vosk_end_original": detection["end"],
            }
        )

    return aligned
