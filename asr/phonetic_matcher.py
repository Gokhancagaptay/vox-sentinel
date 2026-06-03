"""
Katman 1 — Fonetik Güvenlik Ağı

Vosk yanlış transkripsiyon yaptığında (OOV — Out-of-Vocabulary)
bu modül devreye girer. Örnek: Vosk "piç" yerine "hiçbir" döndürürse,
bu iki kelime arasındaki fonetik mesafe hesaplanır.

Türkçe için Soundex/Metaphone yerine Jaro-Winkler algoritması tercih edilir:
- Türkçe fonolojisi Latin alfabesine iyi yansıdığından karakter bazlı
  mesafe metrikleri güvenilir sonuç verir.
- Jaro-Winkler özellikle ortak önek paylaşan kelimelerde daha hassastır.

Bağımlılık: jellyfish kütüphanesi (pip install jellyfish)
"""

import logging
import unicodedata
import jellyfish

from config.settings import PHONETIC_SIMILARITY_THRESHOLD, PHONETIC_LENGTH_DIFF_MAX
from config.banned_words import YASAKLI_KELIMELER

logger = logging.getLogger(__name__)

# Yasaklı kelimeler normalize edilerek önbelleklenir; her çağrıda tekrar hesaplanmaz.
_BANNED_NORMALIZED: list[str] = [
    unicodedata.normalize("NFC", w.lower()) for w in YASAKLI_KELIMELER
]


def find_phonetic_match(
    word: str,
    banned_list: list[str] = YASAKLI_KELIMELER,
    threshold: float = PHONETIC_SIMILARITY_THRESHOLD,
) -> tuple[str | None, float]:
    """
    Tek bir kelimenin yasaklı kelimelerle fonetik benzerliğini ölçer.

    Args:
        word        : Kontrol edilecek kelime (büyük/küçük harf fark etmez).
        banned_list : Karşılaştırılacak yasaklı kelimeler listesi.
        threshold   : Eşleşme için gereken minimum benzerlik skoru (0.0–1.0).

    Returns:
        (eşleşen_yasaklı_kelime, benzerlik_skoru)
        Eşleşme yoksa (None, 0.0) döner.

    Örnekler:
        find_phonetic_match("pis")     → ("piç", 0.89)   ← yakalanır
        find_phonetic_match("hiçbir")  → (None, 0.0)     ← geçer
    """
    word_normalized = unicodedata.normalize("NFC", word.lower().strip())

    # Çok kısa kelimeleri fonetik karşılaştırmaya sokma.
    # "bu", "o", "biz" gibi 2 karakterli kelimeler "budala" ile
    # Jaro-Winkler'da yüksek skor alabilir; bu yanlış alarmları önler.
    if len(word_normalized) < 3:
        return None, 0.0

    # Varsayılan listeyle çağrıldıysa önbelleklenmiş normalized versiyonu kullan.
    if banned_list is YASAKLI_KELIMELER:
        banned_pairs = zip(YASAKLI_KELIMELER, _BANNED_NORMALIZED)
    else:
        banned_pairs = zip(banned_list, (w.lower() for w in banned_list))

    best_match: str | None = None
    best_score: float = 0.0

    for banned, banned_lower in banned_pairs:
        # Kelime uzunlukları çok farklıysa karşılaştırma yapma.
        if abs(len(word_normalized) - len(banned_lower)) > PHONETIC_LENGTH_DIFF_MAX:
            continue

        score = jellyfish.jaro_winkler_similarity(word_normalized, banned_lower)
        if score >= threshold and score > best_score:
            best_match = banned
            best_score = score

    return best_match, best_score


def scan_for_phonetic_matches(
    vosk_words: list[dict],
    banned_list: list[str] = YASAKLI_KELIMELER,
    threshold: float = PHONETIC_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Vosk çıktısındaki tüm kelimeleri fonetik eşleşme için tarar.

    Bu fonksiyon Vosk'un yanlış tanıdığı (OOV) kelimeleri
    fonetik benzerlik üzerinden tespit etmek için tasarlanmıştır.
    Yalnızca eşik değeri geçen kelimeler döndürülür.

    Args:
        vosk_words  : Vosk'tan gelen kelime listesi.
                      Her eleman: {"word", "start", "end", ...}
        banned_list : Yasaklı kelimeler listesi.
        threshold   : Fonetik benzerlik eşiği.

    Returns:
        Eşleşen kelimeler listesi; her eleman aşağıdaki anahtarları içerir:
        - "word"            : Vosk'un tanıdığı orijinal kelime
        - "start"           : Başlangıç zamanı (saniye)
        - "end"             : Bitiş zamanı (saniye)
        - "matched_banned"  : Eşleşen yasaklı kelime
        - "similarity_score": Benzerlik skoru (0.0–1.0)
        - "source"          : "phonetic" (kaynağı belirtir)
    """
    matches: list[dict] = []

    for word_info in vosk_words:
        matched_banned, score = find_phonetic_match(
            word_info["word"], banned_list, threshold
        )
        if matched_banned:
            matches.append({
                "word":             word_info["word"],
                "start":            word_info["start"],
                "end":              word_info["end"],
                "matched_banned":   matched_banned,
                "similarity_score": score,
                "source":           "phonetic",
            })
            logger.debug(
                "[FONETİK] '%s' → '%s' (skor: %.3f)",
                word_info["word"], matched_banned, score,
            )

    return matches
