# Vosk model dizini

Bu depoda **tam Vosk modeli yoktur** (dosyalar çok büyük olduğu için GitHub’a eklenmez). Aşağıdaki adımlarla kendi makinenizde `model/` klasörünü doldurun.

## 1. Model indirin

Resmi model listesi: [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)

Türkçe için örnek paketler (ihtiyaca göre birini seçin):

- `vosk-model-small-tr-0.3` — daha hızlı, daha küçük
- `vosk-model-tr-0.3` — daha büyük, genelde daha iyi doğruluk

## 2. Kurulum

1. İndirdiğiniz **zip** dosyasını açın.
2. İçindeki model klasörünün **içeriğini** (veya tek klasörü) bu `model/` dizinine yerleştirin.

Doğru yapılandırma sonrası `model/` altında tipik olarak `am`, `conf`, `graph`, `ivector` gibi alt klasörler ve çeşitli yapılandırma dosyaları bulunur. `config/settings.py` içindeki `VOSK_MODEL_PATH` varsayılan olarak proje kökündeki `model` klasörünü işaret eder.

## 3. Doğrulama

Proje kökünden:

```bash
python main.py bir_test_kaydi.wav
```

Model yolu yanlışsa veya model eksikse Vosk yükleme aşamasında hata alırsınız.

## Not

Yerel olarak oluşturduğunuz model dosyaları `.gitignore` ile dışlanır; yalnızca bu `README.md` depoya eklenir.
