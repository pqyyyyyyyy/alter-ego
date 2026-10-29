"""语音初筛必须拒绝损坏/削波数据，并保护已有人工转写。"""
import importlib.util
from pathlib import Path
import wave

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_voice_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_voice_dataset", SCRIPT)
dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset)


def write_wav(path, pcm, frames=None):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)


def test_silence_and_clipping_are_not_candidates(tmp_path):
    path = tmp_path / "audio.wav"
    write_wav(path, b"\0\0" * 64000)
    record = dataset.inspect_wav(path)
    assert record["peak"] == 0
    assert not record["candidate"]
    write_wav(path, b"\xff\x7f" * 64000)
    record = dataset.inspect_wav(path)
    assert record["clipping_fraction"] == 1
    assert not record["candidate"]


def test_truncated_wav_is_reported(tmp_path):
    path = tmp_path / "audio.wav"
    write_wav(path, b"\x01\x00" * 64000)
    with path.open("r+b") as wav:
        wav.truncate(100)
    record = dataset.inspect_wav(path)
    assert record["error"]
    assert not record["candidate"]


def test_existing_transcripts_are_never_overwritten(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    progress = output / "review.jsonl"
    progress.write_text("人工台词", encoding="utf-8")
    with pytest.raises(ValueError, match="避免覆盖"):
        dataset.prepare(source, output, 1)
    assert progress.read_text(encoding="utf-8") == "人工台词"


def test_output_cannot_mix_with_original_audio(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="分开"):
        dataset.prepare(source, source / "processed", 1)
