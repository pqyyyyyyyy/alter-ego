"""Optional TTS transport and subtitle-only fallback."""

from __future__ import annotations

import asyncio
import base64
import importlib
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.contracts import DialogTurn  # noqa: E402
from alter_ego.voice.edge_tts import EdgeTTS  # noqa: E402


def test_edge_stream_collects_only_audio():
    async def stream():
        yield {"type": "WordBoundary", "text": "こんにちは"}
        yield {"type": "audio", "data": b"abc"}
        yield {"type": "audio", "data": b"def"}

    calls = []
    tts = EdgeTTS.__new__(EdgeTTS)
    tts.voice = "ja-JP-NanamiNeural"
    tts._edge_tts = SimpleNamespace(Communicate=lambda text, voice: (
        calls.append((text, voice)) or SimpleNamespace(stream=stream)))
    assert asyncio.run(tts.synthesize_mp3("こんにちは")) == b"abcdef"
    assert calls == [("こんにちは", "ja-JP-NanamiNeural")]


def test_speak_audio_and_fallback(monkeypatch):
    server = importlib.import_module("alter_ego.server.app")
    runtime = server.Runtime()
    events = []

    async def send(message):
        events.append(message)

    runtime.hub.send = send
    monkeypatch.setattr(server, "rt", runtime)
    turn = DialogTurn(user_text="hi", speak_ja="こんにちは", subtitle_zh="你好")

    class WorkingTTS:
        async def synthesize_mp3(self, text):
            assert text == "こんにちは"
            return b"ID3"

    runtime.tts = WorkingTTS()
    asyncio.run(server._speak(turn))
    speech = events[-1]
    assert speech["text"] == "你好"
    assert base64.b64decode(speech["audio"].split(",", 1)[1]) == b"ID3"
    assert "mouth" not in speech
    assert runtime.tts_error == ""

    class BrokenTTS:
        async def synthesize_mp3(self, text):
            raise ConnectionError("offline")

    runtime.tts = BrokenTTS()
    asyncio.run(server._speak(turn))
    assert events[-1]["audio"] is None
    assert events[-1]["text"] == "你好"
    assert "offline" in runtime.tts_error


def test_speak_timeout_falls_back(monkeypatch):
    server = importlib.import_module("alter_ego.server.app")
    runtime = server.Runtime()
    events = []

    async def send(message):
        events.append(message)

    class SlowTTS:
        async def synthesize_mp3(self, text):
            raise asyncio.TimeoutError('transport deadline reached')

    runtime.hub.send = send
    runtime.tts = SlowTTS()
    monkeypatch.setattr(server, "rt", runtime)
    asyncio.run(server._speak(DialogTurn(user_text="hi", speak_ja="はい")))
    assert events[-1]["audio"] is None
    assert "TimeoutError" in runtime.tts_error
