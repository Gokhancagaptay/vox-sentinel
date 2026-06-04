import _bootstrap_path  # noqa: F401
import sounddevice as sd

for dev_idx in [14, 15]:
    info = sd.query_devices(dev_idx)
    name = info["name"]
    default_rate = int(info["default_samplerate"])
    print(f"Device {dev_idx}: {name}  default_samplerate={default_rate}")
    for rate in [48000, 44100, 16000, default_rate]:
        try:
            data = sd.rec(
                int(0.5 * rate), samplerate=rate, channels=1, dtype="int16", device=dev_idx
            )
            sd.wait()
            print(f"  rate={rate} OK → {len(data)} samples kaydedildi")
            break
        except Exception as e:
            print(f"  rate={rate} FAIL: {e}")
