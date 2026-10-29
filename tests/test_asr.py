"""无模型单测：PCM 转换、句间隔离、录音格式和缓冲区上限。"""
from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.voice.asr import SenseVoiceASR  # noqa: E402


@pytest.fixture
def recognizer(tmp_path, monkeypatch):
    (tmp_path / "model.int8.onnx").touch()
    (tmp_path / "tokens.txt").touch()
    streams = []

    class Stream:
        result = SimpleNamespace(text=" 你好。 ")

        def accept_waveform(self, rate, samples):
            self.rate, self.samples = rate, samples

    class Recognizer:
        fail = False

        def create_stream(self):
            stream = Stream()
            streams.append(stream)
            return stream

        def decode_stream(self, stream):
            if self.fail:
                raise RuntimeError("decode failed")

    rec = Recognizer()
    calls = []
    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        OfflineRecognizer=SimpleNamespace(from_sense_voice=lambda **kwargs: (
            calls.append(kwargs) or rec)),
    ))
    return SenseVoiceASR(tmp_path), streams, rec, calls


def test_pcm_normalization_and_sentence_isolation(recognizer):
    asr, streams, _, calls = recognizer
    asr.feed(np.array([-32768, 0, 32767], dtype="<i2").tobytes())
    assert asr.poll() == ("你好。", True)
    assert streams[0].rate == 16000
    np.testing.assert_allclose(streams[0].samples, [-1, 0, 32767 / 32768])
    assert streams[0].samples.dtype == np.float32
    assert calls[0]["provider"] == "cpu"
    assert calls[0]["language"] == "zh"
    assert asr.poll() is None
    asr.feed(b"\x01\x00")
    asr.transcribe()
    assert len(streams[1].samples) == 1


def test_reset_and_decode_failure_discard_previous_sentence(recognizer):
    asr, streams, rec, _ = recognizer
    asr.feed(b"\x01\x00")
    asr.reset()
    assert asr.poll() is None
    asr.feed(b"\x01\x00")
    rec.fail = True
    with pytest.raises(RuntimeError):
        asr.transcribe()
    assert asr.poll() is None


def test_invalid_or_oversized_pcm(recognizer):
    asr, _, _, _ = recognizer
    with pytest.raises(ValueError, match="偶数"):
        asr.feed(b"x")
    with pytest.raises(ValueError, match="30 秒"):
        asr.feed(b"\0" * (30 * 16000 * 2 + 2))
    asr.feed(b"")
    assert asr.transcribe() == ""


def test_missing_model_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.int8.onnx"):
        SenseVoiceASR(tmp_path)


@pytest.mark.parametrize("rate,channels,width", [
    (16000, 1, 2), (48000, 1, 2), (16000, 2, 2), (16000, 1, 1),
])
def test_wav_format_validation(tmp_path, rate, channels, width):
    spec = importlib.util.spec_from_file_location("bench_asr", ROOT / "scripts/bench_asr.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    path = tmp_path / "test.wav"
    pcm = b"\0" * (rate * channels * width)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    if (rate, channels, width) == (16000, 1, 2):
        assert bench.read_pcm(path) == pcm
    else:
        with pytest.raises(ValueError, match="16000 Hz"):
            bench.read_pcm(path)
