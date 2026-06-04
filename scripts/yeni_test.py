"""Whisper API verbose_json kelime zamanları denemesi (OPENAI_API_KEY gerekir)."""
import _bootstrap_path  # noqa: F401
import openai

ROOT = _bootstrap_path.PROJECT_ROOT

client = openai.OpenAI()

audio_path = ROOT / "audio.mp3"
with open(audio_path, "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

for word_info in result.words:
    print(word_info.word, word_info.start, word_info.end)
