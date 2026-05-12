"""
Merkezi yasaklı kelime sözlüğü.
Tüm ASR katmanları ve fonetik eşleştirici bu listeden beslenir.

KURAL: Yalnızca KÖK kelimeleri ekleyin.
  - Türevler, ekler, çoğullar OTOMATİK yakalanır (alt-dize kontrolü).
  - Fonetik benzerler OTOMATİK yakalanır (Jaro-Winkler).
  Örnek: "sik" eklenince → siki, sikiş, sikik, sikerim, sikeyim hepsi yakalanır.
"""

YASAKLI_KELIMELER: list[str] = [

    # ══════════════════════════════════════════════════════════════
    # KATEGORİ 1 — Ağır cinsel/bedensel küfürler
    # ══════════════════════════════════════════════════════════════
    "piç",          # piçoş, piçlik, piçler → alt-dize ile yakalanır
    "sik",          # siki, sikiş, sikik, sikerim, sikeyim, siktiret
    "orospu",       # orosbuçuk, orospular, orospulaştı
    "göt",          # götlek, götveren, götlük
    "amına",        # amk, amcık kökleriyle ilgili
    "amk",
    "amcık",
    "yarrak",       # yarrağı, yarrağına
    "dalyarak",
    "taşak",        # taşağına
    "oç",
    "orosbuçuk",

    # ══════════════════════════════════════════════════════════════
    # KATEGORİ 2 — Hakaret / aşağılama
    # ══════════════════════════════════════════════════════════════
    "kahpe",        # kahpeler, kahpelik
    "ibne",         # ibneler, ibnelik
    "bok",          # boktan, boklu, bokluk
    # NOT: "it" eklenmedi — 2 karakter, çok sık yanlış alarm verir
    "şerefsiz",
    "namussuz",
    "alçak",
    "aşağılık",
    "rezil",
    "haysiyetsiz",
    "şıllık",       # argo hakaret
    "pezevenk",     # pezevenk → türevler
    "kereste",      # argo hakaret
    "sürtük",
    "dangalak",

    # ══════════════════════════════════════════════════════════════
    # KATEGORİ 3 — Hafif hakaret / alay
    # ══════════════════════════════════════════════════════════════
    "aptal",        # aptallık, aptallar
    "salak",        # salaklık
    "gerizekalı",
    # NOT: "mal" eklenmedi — malzeme, maliyet gibi kelimelerle çakışır
    "ahmak",
    "budala",
    "serseri",
    "ezik",         # ezikler, eziklik
    "avanak",
    "manyak",       # manyaklar, manyaklık
    # NOT: "deli" eklenmedi — delil, delik gibi masum kelimelerle çakışır
    "saloz",        # kaba argo
    "hödük",
    "göbelek",      # argo
]
