"""
Pipeline Entegrasyon Testleri

core/pipeline.py'nin üç katmanlı akışını uçtan uca test eder.
Model bağımlılıkları (Vosk, Whisper) mock'lanır;
karar motoru ve fonetik eşleştirici gerçek kodla çalışır.

Patch hedefleri: core.pipeline.* namespace'i
  - get_word_timestamps      → Vosk modelini atla
  - _resolve_whisper_transcriber → Whisper modelini atla
  - to_vosk_wav             → Format dönüşümünü atla
  - cleanup_temp_wav        → Dosya silme işlemini atla
  - apply_censor_beeps      → FFmpeg ses işlemini atla
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from core.pipeline import PipelineResult, run_censorship_pipeline


@contextmanager
def _pipeline_mocks(vosk_words, whisper_words, beep_output="output.wav"):
    """Standart 5 mock'u tek context manager altında yönetir."""
    whisper_fn = MagicMock(return_value=whisper_words)
    with (
        patch("core.pipeline.get_word_timestamps", return_value=vosk_words),
        patch("core.pipeline._resolve_whisper_transcriber", return_value=(whisper_fn, "local")),
        patch("core.pipeline.to_vosk_wav", return_value=("fake.wav", False)),
        patch("core.pipeline.cleanup_temp_wav"),
        patch("core.pipeline.apply_censor_beeps", return_value=beep_output) as mock_beep,
    ):
        yield mock_beep, whisper_fn


class TestCleanAudio:
    """Yasaklı kelime içermeyen ses — sansür uygulanmamalı."""

    def test_censored_false(self, fake_vosk_words, fake_whisper_words):
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert result.censored is False

    def test_no_censor_segments(self, fake_vosk_words, fake_whisper_words):
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert result.final_censor_segments == []

    def test_apply_beeps_not_called(self, fake_vosk_words, fake_whisper_words):
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words) as (mock_beep, _):
            run_censorship_pipeline("input.wav", "output.wav")
        mock_beep.assert_not_called()

    def test_pipeline_result_type(self, fake_vosk_words, fake_whisper_words):
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert isinstance(result, PipelineResult)


class TestWhisperBannedWord:
    """Whisper yasaklı kelime tespit edince sansür uygulanmalı."""

    def test_censored_true(self, fake_vosk_words, banned_whisper_words):
        with _pipeline_mocks(fake_vosk_words, banned_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert result.censored is True

    def test_censor_segments_not_empty(self, fake_vosk_words, banned_whisper_words):
        with _pipeline_mocks(fake_vosk_words, banned_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert len(result.final_censor_segments) >= 1

    def test_apply_beeps_called_once(self, fake_vosk_words, banned_whisper_words):
        with _pipeline_mocks(fake_vosk_words, banned_whisper_words) as (mock_beep, _):
            run_censorship_pipeline("input.wav", "output.wav")
        mock_beep.assert_called_once()

    def test_whisper_detections_populated(self, fake_vosk_words, banned_whisper_words):
        with _pipeline_mocks(fake_vosk_words, banned_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert len(result.whisper_detections) >= 1
        assert result.whisper_detections[0]["matched_banned"] == "aptal"

    def test_censor_segment_has_required_keys(self, fake_vosk_words, banned_whisper_words):
        with _pipeline_mocks(fake_vosk_words, banned_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        seg = result.final_censor_segments[0]
        assert "start_ms" in seg
        assert "end_ms" in seg
        assert seg["start_ms"] < seg["end_ms"]


class TestPhoneticOnlyDetection:
    """Whisper kapalı, sadece Vosk + fonetik tespit."""

    def test_phonetic_detections_populated(self, phonetic_vosk_words):
        vosk_fn = MagicMock(return_value=phonetic_vosk_words)
        with (
            patch("core.pipeline.get_word_timestamps", side_effect=vosk_fn),
            patch("core.pipeline.to_vosk_wav", return_value=("fake.wav", False)),
            patch("core.pipeline.cleanup_temp_wav"),
            patch("core.pipeline.apply_censor_beeps", return_value="output.wav"),
        ):
            result = run_censorship_pipeline(
                "input.wav", "output.wav",
                use_whisper=False, use_phonetic=True,
            )
        assert len(result.phonetic_detections) >= 1

    def test_phonetic_match_is_aptal(self, phonetic_vosk_words):
        with (
            patch("core.pipeline.get_word_timestamps", return_value=phonetic_vosk_words),
            patch("core.pipeline.to_vosk_wav", return_value=("fake.wav", False)),
            patch("core.pipeline.cleanup_temp_wav"),
            patch("core.pipeline.apply_censor_beeps", return_value="output.wav"),
        ):
            result = run_censorship_pipeline(
                "input.wav", "output.wav",
                use_whisper=False, use_phonetic=True,
            )
        matched = {d["matched_banned"] for d in result.phonetic_detections}
        assert "aptal" in matched

    def test_censored_true_when_phonetic_match(self, phonetic_vosk_words):
        with (
            patch("core.pipeline.get_word_timestamps", return_value=phonetic_vosk_words),
            patch("core.pipeline.to_vosk_wav", return_value=("fake.wav", False)),
            patch("core.pipeline.cleanup_temp_wav"),
            patch("core.pipeline.apply_censor_beeps", return_value="output.wav"),
        ):
            result = run_censorship_pipeline(
                "input.wav", "output.wav",
                use_whisper=False, use_phonetic=True,
            )
        assert result.censored is True


class TestPipelineResultFields:
    """PipelineResult alanlarının doğru doldurulduğunu doğrular."""

    def test_vosk_words_preserved(self, fake_vosk_words, fake_whisper_words):
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert result.vosk_words == fake_vosk_words

    def test_whisper_words_preserved(self, fake_vosk_words, fake_whisper_words):
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert result.whisper_words == fake_whisper_words

    def test_anchors_found_when_words_overlap(self, fake_vosk_words, fake_whisper_words):
        # "merhaba" ve "dünya" her iki listede var → anchor bulunmalı
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert result.anchor_count > 0

    def test_output_file_set_when_censored(self, fake_vosk_words, banned_whisper_words):
        with _pipeline_mocks(fake_vosk_words, banned_whisper_words, beep_output="out.wav"):
            result = run_censorship_pipeline("input.wav", "out.wav")
        assert result.output_file == "out.wav"

    def test_output_file_none_when_clean(self, fake_vosk_words, fake_whisper_words):
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words):
            result = run_censorship_pipeline("input.wav", "output.wav")
        assert result.output_file is None

    def test_whisper_fn_called_with_audio_path(self, fake_vosk_words, fake_whisper_words):
        with _pipeline_mocks(fake_vosk_words, fake_whisper_words) as (_, whisper_fn):
            run_censorship_pipeline("my_audio.wav", "output.wav")
        whisper_fn.assert_called_once_with("my_audio.wav")
