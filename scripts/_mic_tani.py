"""Windows mikrofon erişim engelinin nedenini tespit eder."""
import _bootstrap_path  # noqa: F401

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import winreg

PRIVACY_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"


def reg_oku(hive, key_yolu, deger_adi="Value"):
    try:
        with winreg.OpenKey(hive, key_yolu) as k:
            val, _ = winreg.QueryValueEx(k, deger_adi)
            return val
    except FileNotFoundError:
        return "(Bulunamadı)"
    except Exception as e:
        return f"(Hata: {e})"


print("=" * 60)
print("  Windows Mikrofon Gizlilik Ayarları Tanısı")
print("=" * 60)

sistem_izni = reg_oku(winreg.HKEY_LOCAL_MACHINE, PRIVACY_KEY)
print(f"\n[1] Sistem geneli mikrofon izni : {sistem_izni}")
print("    ('Allow' = açık, 'Deny' = kapalı)\n")

kullanici_izni = reg_oku(winreg.HKEY_CURRENT_USER, PRIVACY_KEY)
print(f"[2] Kullanıcı düzeyi izin        : {kullanici_izni}\n")

python_yolu = sys.executable
print(f"[3] Python yürütülebilir         : {python_yolu}")

is_store = "WindowsApps" in python_yolu or "PythonSoftwareFoundation" in python_yolu
print(f"[4] Microsoft Store Python?      : {'EVET ← bu sorun olabilir!' if is_store else 'Hayır'}\n")

if is_store:
    store_key = PRIVACY_KEY + r"\NonPackaged"
    store_izni = reg_oku(winreg.HKEY_CURRENT_USER, store_key)
    print(f"[5] Mağaza dışı (desktop) izni  : {store_izni}")
    print("    (Masaüstü uygulamalar için 'Allow' olmalı)\n")

print("=" * 60)
print("\nÖNERİLER:")
print()

if kullanici_izni != "Allow":
    print("  [!] Kullanıcı mikrofon izni KAPALI.")
    print("      Düzeltme: Ayarlar > Gizlilik > Mikrofon > 'Açık' yapın")
    print("      Veya şu komutu PowerShell (Yönetici) ile çalıştırın:")
    print(f'      reg add "HKCU\\{PRIVACY_KEY}" /v Value /t REG_SZ /d Allow /f')
    print()

if sistem_izni != "Allow":
    print("  [!] Sistem geneli mikrofon izni KAPALI.")
    print("      Düzeltme: Ayarlar > Gizlilik > Mikrofon > En üstteki toggle = Açık")
    print()

if is_store:
    print("  [!] Microsoft Store Python kullanıyorsunuz.")
    print("      Bu sürüm sandboxed çalışır ve mikrofona erişemeyebilir.")
    print()
    print("  KALICI ÇÖZÜM: Python'u doğrudan python.org'dan kurun:")
    print("    https://www.python.org/downloads/")
    print("    Kurarken 'Add Python to PATH' seçeneğini işaretleyin")
    print()
    print("  HIZLI ÇÖZÜM DENEMESİ: PowerShell (Yönetici) ile:")
    print(f'    reg add "HKCU\\{PRIVACY_KEY}" /v Value /t REG_SZ /d Allow /f')
    print(f'    reg add "HKCU\\{PRIVACY_KEY}\\NonPackaged" /v Value /t REG_SZ /d Allow /f')
    print()

if kullanici_izni == "Allow" and sistem_izni == "Allow" and not is_store:
    print("  [OK] İzinler görünürde açık ama yine de hata alınıyor.")
    print("       Ses sürücüsünü güncelleyin veya bilgisayarı yeniden başlatın.")
