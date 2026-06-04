"""
Testler: config/banned_words.py
- YASAKLI_KELIMELER liste bütünlüğü
- YASAKLI_AGIRLIKLI seviye tutarlılığı
- agirlik_sayaci() doğruluğu
"""
import pytest

from config.banned_words import (
    YASAKLI_AGIRLIKLI,
    YASAKLI_KELIMELER,
    agirlik_sayaci,
)

VALID_LEVELS = {"yuksek", "orta", "dusuk"}


def test_liste_bos_degil():
    assert len(YASAKLI_KELIMELER) > 0


def test_agirlikli_ve_liste_eslesiyor():
    """Düz liste ile ağırlıklı dict tamamen örtüşmeli."""
    assert set(YASAKLI_KELIMELER) == set(YASAKLI_AGIRLIKLI.keys())


def test_tum_seviyeler_gecerli():
    for kelime, seviye in YASAKLI_AGIRLIKLI.items():
        assert seviye in VALID_LEVELS, f"'{kelime}' geçersiz seviye: '{seviye}'"


def test_yuksek_kategori_var():
    yuksekler = [k for k, v in YASAKLI_AGIRLIKLI.items() if v == "yuksek"]
    assert len(yuksekler) > 0


def test_agirlik_sayaci_bos():
    assert agirlik_sayaci([]) == {"yuksek": 0, "orta": 0, "dusuk": 0}


def test_agirlik_sayaci_karisik():
    tespitler = [
        {"matched_banned": "sik"},       # yuksek
        {"matched_banned": "piç"},       # yuksek
        {"matched_banned": "kahpe"},     # orta
        {"matched_banned": "aptal"},     # dusuk
        {"matched_banned": "aptal"},     # dusuk (tekrar)
    ]
    sayac = agirlik_sayaci(tespitler)
    assert sayac["yuksek"] == 2
    assert sayac["orta"] == 1
    assert sayac["dusuk"] == 2


def test_agirlik_sayaci_bilinmeyen_kelime():
    """Listede olmayan kelime → 'dusuk' varsayılan."""
    tespitler = [{"matched_banned": "bilmiyorum_bu_kelime"}]
    sayac = agirlik_sayaci(tespitler)
    assert sayac["dusuk"] == 1


def test_liste_duplikat_yok():
    assert len(YASAKLI_KELIMELER) == len(set(YASAKLI_KELIMELER))


@pytest.mark.parametrize("kelime", ["sik", "piç", "orospu", "göt", "amk"])
def test_agir_kelimeler_mevcut(kelime):
    assert kelime in YASAKLI_AGIRLIKLI
    assert YASAKLI_AGIRLIKLI[kelime] == "yuksek"
