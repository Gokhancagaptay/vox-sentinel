"""
Katman 2 — Çok-Kaynaklı Oylama Motoru

Diyagramdaki oylama mantığı:
    Whisper eşleşti       → kes (yüksek güven)
    Fonetik eşleşti       → kes (orta güven)
    İkisi de hayır        → geç

Bu modül iki kaynaktan gelen tespitleri birleştirir ve
çakışan/yakın zaman aralıklarını tek bir sansür segmentinde mergeler.
"""

import unicodedata
import logging
import jellyfish

from config.settings import (
    VOTE_MERGE_THRESHOLD_MS,
    PHONETIC_SIMILARITY_THRESHOLD,
    PHONETIC_LENGTH_DIFF_MAX,
    BIGRAM_MAX_GAP_SEC,
)
from config.banned_words import YASAKLI_KELIMELER
from config.whitelist import beyaz_listede_mi

logger = logging.getLogger(__name__)

# Whisper tespitinde kullanılan benzerlik eşikleri.
# Kısa kelimelerde Jaro-Winkler çok geniş eşleşme yapar:
#   jaro_winkler("siz", "sik") = 0.82  → "siz" (Türkçe "siz/you") yanlış alarm!
# Bu nedenle kelime uzunluğuna göre kademeli eşik uygulanır.
WHISPER_FUZZY_THRESHOLD_SHORT  = 1.01  # 3 karakter veya altı → fuzzy KAPALI (sadece tam eşleşme)
WHISPER_FUZZY_THRESHOLD_MEDIUM = 0.88  # 4-5 karakter          → yüksek eşik
WHISPER_FUZZY_THRESHOLD_LONG   = 0.80  # 6+ karakter           → normal eşik

# Bigram kontrolünde kullanılan eşik.
# "buadamı" ↔ "budala" gibi yanlış bigramleri önlemek için
# tek kelime eşiğinden daha yüksek tutulur.
WHISPER_BIGRAM_FUZZY_THRESHOLD = 0.93


def _fuzzy_threshold_for(word: str) -> float:
    """Kelime uzunluğuna göre uygun Jaro-Winkler eşiğini döndürür."""
    length = len(word)
    if length <= 3:
        return WHISPER_FUZZY_THRESHOLD_SHORT   # Etkin olarak kapalı
    if length <= 5:
        return WHISPER_FUZZY_THRESHOLD_MEDIUM
    return WHISPER_FUZZY_THRESHOLD_LONG


def _normalize(text: str) -> str:
    """NFC Unicode normalizasyonu + küçük harf + baştaki/sondaki boşluk temizleme."""
    return unicodedata.normalize("NFC", text.lower().strip())


def find_whisper_banned_words(whisper_words: list[dict]) -> list[dict]:
    """
    Whisper transkript çıktısında yasaklı kelimeleri arar.

    Üç aşamalı eşleştirme:
    1. Tek kelime: Tam/alt dize eşleşmesi
    2. Tek kelime: Jaro-Winkler fonetik benzerliği
       Örnek: Whisper "piç" yerine "piş" derse → benzerlik ~0.82 → yakalanır
    3. Bigram (ikili kelime): Whisper'ın uzun kelimeleri bölmesini yakalar
       Örnek: "orospu" → ["Oros", "bu"] → "orosbu" fonetik olarak "orospu" ile eşleşir

    Args:
        whisper_words : Whisper'dan gelen kelime listesi.

    Returns:
        Tespit edilen her yasaklı kelime için:
        - "word"           : Orijinal transkript kelimesi
        - "start"          : Başlangıç zamanı (saniye)
        - "end"            : Bitiş zamanı (saniye)
        - "matched_banned" : Eşleşen yasaklı kelime
        - "source"         : "whisper"
    """
    detections: list[dict] = []

    # ── Aşama 1 & 2: Tek kelime kontrolü ─────────────────────────
    for word_info in whisper_words:
        word_normalized = _normalize(word_info["word"])
        matched_banned, best_score = _match_single_word(word_normalized)

        if matched_banned:
            detections.append({
                "word":           word_info["word"],
                "start":          word_info["start"],
                "end":            word_info["end"],
                "matched_banned": matched_banned,
                "source":         "whisper",
            })
            logger.debug(
                "[WHISPER TESPIT] '%s' → '%s' (skor: %.3f)",
                word_info["word"], matched_banned, best_score,
            )

    # ── Aşama 3: Bigram kontrolü (ardışık iki kelimeyi birleştir) ─
    # "Oros" + "bu" = "Orosbu" → "orospu" ile fonetik eşleşme
    bigram_detections = _check_bigrams(whisper_words)
    detections.extend(bigram_detections)

    # ── Tekrar (dedup): aynı start/end aralığı birden fazla kez raporlandıysa ilkini tut ──
    seen: set[tuple[float, float]] = set()
    unique: list[dict] = []
    for det in detections:
        key = (det["start"], det["end"])
        if key not in seen:
            seen.add(key)
            unique.append(det)

    return unique


def _match_single_word(word_normalized: str) -> tuple[str | None, float]:
    """
    Tek bir kelimeyi yasaklı liste ile karşılaştırır.
    (word, skor) döndürür; eşleşme yoksa (None, 0.0)

    Beyaz listede olan kelimeler hiçbir koşulda eşleşmez.
    """
    # Beyaz listede varsa anında geç (tam eşleşme veya önek kuralı)
    if beyaz_listede_mi(word_normalized):
        return None, 0.0

    best_match: str | None = None
    best_score: float = 0.0

    for banned in YASAKLI_KELIMELER:
        banned_normalized = _normalize(banned)

        # Tam / alt dize
        if banned_normalized in word_normalized:
            return banned, 1.0

        # Fonetik benzerlik — çok farklı uzunluklarda karşılaştırma yapma
        # Kısa kelimeler için eşik çok yüksek tutulur (yanlış alarm önleme)
        if abs(len(word_normalized) - len(banned_normalized)) <= PHONETIC_LENGTH_DIFF_MAX:
            threshold = _fuzzy_threshold_for(word_normalized)
            score = jellyfish.jaro_winkler_similarity(word_normalized, banned_normalized)
            if score >= threshold and score > best_score:
                best_match = banned
                best_score = score

    return best_match, best_score


def _check_bigrams(whisper_words: list[dict]) -> list[dict]:
    """
    Ardışık iki Whisper kelimesini birleştirerek yasaklı kelime arar.
    Whisper'ın uzun kelimeleri (orospu, pezevenk…) bazen ikiye bölmesini yakalar.

    Birleştirilen çift, bireysel tespitle çakışırsa raporlanmaz.
    Zaman damgası olarak iki kelimenin toplam aralığı kullanılır.

    Not: Bigram eşiği (WHISPER_BIGRAM_FUZZY_THRESHOLD) tek kelime eşiğinden
    yüksek tutulur. "buadamı" ↔ "budala" gibi yanlış eşleşmeler bu sayede elenir.
    """
    bigram_detections: list[dict] = []

    for i in range(len(whisper_words) - 1):
        w1 = whisper_words[i]
        w2 = whisper_words[i + 1]

        # İki kelime arasında en fazla BIGRAM_MAX_GAP_SEC boşluk olsun
        if w2["start"] - w1["end"] > BIGRAM_MAX_GAP_SEC:
            continue

        combined = _normalize(w1["word"]) + _normalize(w2["word"])

        # Birleşik kelime beyaz listede varsa atla
        if beyaz_listede_mi(combined):
            continue

        # Yalnızca tam/alt dize eşleşmesini veya çok yüksek fonetik skoru kabul et
        matched_banned: str | None = None
        best_score: float = 0.0

        for banned in YASAKLI_KELIMELER:
            banned_normalized = _normalize(banned)

            # Kesin alt dize eşleşmesi (en güvenilir)
            if banned_normalized in combined:
                matched_banned = banned
                best_score = 1.0
                break

            # Uzunluk farkı fazlaysa fonetik karşılaştırma yapma
            if abs(len(combined) - len(banned_normalized)) > PHONETIC_LENGTH_DIFF_MAX:
                continue

            score = jellyfish.jaro_winkler_similarity(combined, banned_normalized)
            if score >= WHISPER_BIGRAM_FUZZY_THRESHOLD and score > best_score:
                matched_banned = banned
                best_score = score

        if matched_banned:
            bigram_detections.append({
                "word":           f"{w1['word']} {w2['word']}",
                "start":          w1["start"],
                "end":            w2["end"],
                "matched_banned": matched_banned,
                "source":         "whisper-bigram",
            })
            logger.debug(
                "[WHISPER BIGRAM] '%s %s' → '%s' (skor: %.3f)",
                w1["word"], w2["word"], matched_banned, best_score,
            )

    return bigram_detections


def vote_and_merge(
    whisper_detections: list[dict],
    phonetic_detections: list[dict],
    merge_threshold_ms: int = VOTE_MERGE_THRESHOLD_MS,
) -> list[dict]:
    """
    Whisper ve fonetik tespitleri birleştirerek final sansür listesini üretir.

    Oylama kuralları (diyagramdan):
    - Whisper tespit etti → KES
    - Fonetik tespit etti → KES
    - İkisi de hayır      → GEÇER

    Birleştirme: merge_threshold_ms içindeki iki segment tek segmentte birleşir.
    Bu, aynı küfürü farklı kaynaklardan tespit eden aralıkların
    tek sansür bloğuna dönüşmesini sağlar.

    Args:
        whisper_detections  : whisper motorundan gelen tespitler.
        phonetic_detections : fonetik eşleştirmeden gelen tespitler.
        merge_threshold_ms  : Bu kadar ms içindeki tespitler birleştirilir.

    Returns:
        Final sansür segmentleri listesi. Her eleman:
        - "start_ms"       : Segment başlangıcı (milisaniye)
        - "end_ms"         : Segment bitişi (milisaniye)
        - "word"           : Tespit edilen orijinal kelime
        - "matched_banned" : Eşleşen yasaklı kelime
        - "source"         : Kaynak bilgisi ("whisper", "phonetic" veya ikisi)
    """
    all_raw: list[dict] = []

    # Her iki kaynağı da saniyeden milisaniyeye çevirerek tek listeye ekle
    for detection in whisper_detections:
        all_raw.append({
            "start_ms":      int(detection["start"] * 1000),
            "end_ms":        int(detection["end"]   * 1000),
            "word":          detection.get("word", ""),
            "matched_banned": detection.get("matched_banned", ""),
            "source":        detection["source"],
        })

    for detection in phonetic_detections:
        all_raw.append({
            "start_ms":      int(detection["start"] * 1000),
            "end_ms":        int(detection["end"]   * 1000),
            "word":          detection.get("word", ""),
            "matched_banned": detection.get("matched_banned", ""),
            "source":        detection["source"],
        })

    if not all_raw:
        return []

    # Zamana göre sırala
    all_raw.sort(key=lambda seg: seg["start_ms"])

    # Yakın/çakışan segmentleri birleştir
    merged: list[dict] = [all_raw[0]]

    for current in all_raw[1:]:
        last = merged[-1]

        if current["start_ms"] <= last["end_ms"] + merge_threshold_ms:
            # Segment genişlet ve kaynakları birleştir
            last["end_ms"] = max(last["end_ms"], current["end_ms"])
            existing_sources = set(last["source"].split("+"))
            existing_sources.add(current["source"])
            last["source"] = "+".join(sorted(existing_sources))
        else:
            merged.append(current)

    logger.info(
        "[OYLAMA] %d Whisper + %d fonetik tespit → %d final segment.",
        len(whisper_detections), len(phonetic_detections), len(merged),
    )
    return merged
