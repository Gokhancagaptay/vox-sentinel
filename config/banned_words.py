"""
Merkezi yasaklı kelime sözlüğü.
Tüm ASR katmanları ve fonetik eşleştirici bu listeden beslenir.

KURAL: Yalnızca KÖK kelimeleri ekleyin.
  - Türevler, ekler, çoğullar OTOMATİK yakalanır (alt-dize kontrolü).
  - Fonetik benzerler OTOMATİK yakalanır (Jaro-Winkler).
  Örnek: "sik" eklenince → siki, sikiş, sikik, sikerim, sikeyim hepsi yakalanır.

Ağırlık seviyeleri:
  "yuksek" — Ağır cinsel/bedensel küfürler (her koşulda sansür)
  "orta"   — Hakaret / aşağılama
  "dusuk"  — Hafif hakaret / alay

YASAKLI_KELIMELER: geriye dönük uyumlu düz liste (mevcut pipeline kullanır)
YASAKLI_AGIRLIKLI: {kelime: seviye} dict — raporlama ve öncelik için
"""

# Ağırlıklı sözlük: {kök_kelime: seviye}
YASAKLI_AGIRLIKLI: dict[str, str] = {
    # ── Ağır cinsel/bedensel küfürler ─────────────────────────────
    "piç": "yuksek",  # piçoş, piçlik, piçler
    "sik": "yuksek",  # siki, sikiş, sikik, sikerim, sikeyim
    "orospu": "yuksek",  # orosbuçuk, orospular
    "göt": "yuksek",  # götlek, götveren, götlük
    "amına": "yuksek",  # amk, amcık kökleriyle ilgili
    "amk": "yuksek",
    "amcık": "yuksek",
    "yarrak": "yuksek",  # yarrağı, yarrağına
    "dalyarak": "yuksek",
    "taşak": "yuksek",  # taşağına
    "oç": "yuksek",
    "orosbuçuk": "yuksek",
    # ── Hakaret / aşağılama ────────────────────────────────────────
    # NOT: "it" eklenmedi — 2 karakter, çok sık yanlış alarm verir
    "kahpe": "orta",  # kahpeler, kahpelik
    "ibne": "orta",  # ibneler, ibnelik
    "bok": "orta",  # boktan, boklu, bokluk
    "şerefsiz": "orta",
    "namussuz": "orta",
    "alçak": "orta",
    "aşağılık": "orta",
    "rezil": "orta",
    "haysiyetsiz": "orta",
    "şıllık": "orta",  # argo hakaret
    "pezevenk": "orta",
    "kereste": "orta",  # argo hakaret
    "sürtük": "orta",
    "dangalak": "orta",
    # ── Hafif hakaret / alay ───────────────────────────────────────
    # NOT: "mal" eklenmedi — malzeme, maliyet gibi kelimelerle çakışır
    # NOT: "deli" eklenmedi — delil, delik gibi masum kelimelerle çakışır
    "aptal": "dusuk",  # aptallık, aptallar
    "salak": "dusuk",  # salaklık
    "gerizekalı": "dusuk",
    "ahmak": "dusuk",
    "budala": "dusuk",
    "serseri": "dusuk",
    "ezik": "dusuk",  # ezikler, eziklik
    "avanak": "dusuk",
    "manyak": "dusuk",  # manyaklar, manyaklık
    "saloz": "dusuk",  # kaba argo
    "hödük": "dusuk",
    "göbelek": "dusuk",  # argo
}

# Geriye dönük uyumlu düz liste — mevcut pipeline bu listeyi kullanır
YASAKLI_KELIMELER: list[str] = list(YASAKLI_AGIRLIKLI.keys())


def agirlik_sayaci(tespitler: list[dict]) -> dict[str, int]:
    """
    Tespit listesindeki kelimeleri seviyeye göre say.

    Args:
        tespitler: voting_engine veya phonetic_matcher çıktısı.
                   Her eleman {"matched_banned": ...} içermeli.

    Returns:
        {"yuksek": n, "orta": n, "dusuk": n}
    """
    sayac: dict[str, int] = {"yuksek": 0, "orta": 0, "dusuk": 0}
    for det in tespitler:
        kelime = det.get("matched_banned", "")
        seviye = YASAKLI_AGIRLIKLI.get(kelime, "dusuk")
        sayac[seviye] = sayac.get(seviye, 0) + 1
    return sayac
