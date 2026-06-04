"""
Beyaz Liste — Yanlış Alarm Önleme

İki katmanlı koruma sistemi:

1. BEYAZ_LISTE (tam eşleşme): Tam olarak bu kelimeler tespit edilirse geçirilir.
2. BEYAZ_LISTE_ONEKLER (önek eşleşme): Bu öneklerle başlayan HER kelime korunur.
   Böylece fiil çekimlerini tek tek yazmak gerekmez.
   Örnek: "götür" öneki → götürürken, götürdüm, götürüyor hepsi korunur.
"""
import unicodedata


def _n(t: str) -> str:
    return unicodedata.normalize("NFC", t.lower().strip())


# ─── Tam eşleşme beyaz listesi ────────────────────────────────────
BEYAZ_LISTE: set[str] = {
    # Çok yaygın Türkçe kelimeler — kısa oldukları için fonetik yanlış alarm riski
    "bu", "o", "biz", "siz", "ben", "sen",
    "bir", "ile", "ama", "da", "de",

    # Test sonuçlarından türetilen yanlış alarm kelimeleri
    "bak", "bakın", "bakıyorum", "bakıyor",   # bak ≈ bok (0.80)
    "başka", "başkası", "başkasına",           # başka ≈ taşak (0.78)
    "mantık", "mantıklı", "mantıksız",        # mantık ≈ manyak (0.84)

    # "sik" kökü — bilim/coğrafya terimleri
    "siklon",
    "siklotron",
    "siklus",

    # "sık" fonetik benzerlik riski
    "sıkıntı",
    "sıkıntısı",
    "sıkıntıyla",
    "sıkıştırmak",
}

# ─── Önek bazlı beyaz liste (bu önekle başlayan her kelime korunur) ─
BEYAZ_LISTE_ONEKLER: tuple[str, ...] = (
    # "götür-" fiil kökü: götürmek, götürürken, götürdüm, götürülmüş…
    _n("götür"),
    # "sıkı-" ile başlayan masum kelimeler
    _n("sıkı"),
    # "siklo-" bilim terimi öneki
    _n("siklo"),
)


def beyaz_listede_mi(word_normalized: str) -> bool:
    """
    Kelimenin tam eşleşme veya önek kuralına göre beyaz listede
    olup olmadığını kontrol eder.
    """
    # Tam eşleşme kontrolü
    if word_normalized in {_n(w) for w in BEYAZ_LISTE}:
        return True
    # Önek kontrolü
    return any(word_normalized.startswith(prefix) for prefix in BEYAZ_LISTE_ONEKLER)
