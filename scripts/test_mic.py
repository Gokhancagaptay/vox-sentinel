"""PyAudio ile giriş cihazlarını listeler."""
import _bootstrap_path  # noqa: F401
import pyaudio

p = pyaudio.PyAudio()

print("\n--- SİSTEMDEKİ SES GİRİŞ CİHAZLARI ---")
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info.get("maxInputChannels") > 0:
        print(f"ID: {i} - İsim: {dev_info.get('name')}")
print("--------------------------------------\n")

p.terminate()
