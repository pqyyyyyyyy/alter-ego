"""WAV transport validation and existing browser speech integration."""

import asyncio
import base64
import importlib
import io
import json
import sys
import wave
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alter_ego.contracts import DialogTurn
from alter_ego.voice.gpt_sovits import RemoteGPTSoVITS


def make_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(32000)
        wav.writeframes(b"\x01\x00" * 320)
    return output.getvalue()


def mock_response(monkeypatch, payload, mime="audio/wav"):
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = Message()
    response.headers["Content-Type"] = mime
    response.read.return_value = payload
    calls = []

    def open_url(request, timeout):
        calls.append((request, timeout))
        return response

    monkeypatch.setattr("alter_ego.voice.gpt_sovits.urllib.request.urlopen", open_url)
    return calls


def test_remote_wav_reaches_browser(monkeypatch):
    payload = make_wav()
    calls = mock_response(monkeypatch, payload)
    server = importlib.import_module("alter_ego.server.app")
    runtime = server.Runtime()
    runtime.tts = RemoteGPTSoVITS("http://127.0.0.1:9881/tts")
    events = []

    async def send(message):
        events.append(message)

    runtime.hub.send = send
    monkeypatch.setattr(server, "rt", runtime)
    asyncio.run(server._speak(DialogTurn(user_text="hello", speak_ja="こんにちは", subtitle_zh="你好")))
    audio = events[-1]["audio"]
    assert audio.startswith("data:audio/wav;base64,")
    assert base64.b64decode(audio.split(",", 1)[1]) == payload
    assert json.loads(calls[0][0].data) == {"text": "こんにちは"}
    assert calls[0][1] == 90
    assert runtime.tts_error == ""


@pytest.mark.parametrize(
    "payload,mime",
    [(b'{"error":"failed"}', "application/json"), (b"invalid", "audio/wav"), (b"x" * (2 * 1024 * 1024 + 1), "audio/wav")],
    ids=["json-error", "invalid-wav", "oversize-wav"],
)
def test_bad_audio_preserves_subtitle(monkeypatch, payload, mime):
    mock_response(monkeypatch, payload, mime)
    server = importlib.import_module("alter_ego.server.app")
    runtime = server.Runtime()
    runtime.tts = RemoteGPTSoVITS("http://127.0.0.1:9881/tts")
    events = []

    async def send(message):
        events.append(message)

    runtime.hub.send = send
    monkeypatch.setattr(server, "rt", runtime)
    asyncio.run(server._speak(DialogTurn(user_text="hello", speak_ja="こんにちは", subtitle_zh="你好")))
    assert events[-1]["audio"] is None
    assert events[-1]["text"] == "你好"
    assert runtime.tts_error


@pytest.mark.parametrize('ready,busy', [(True, False), (True, True), (False, False)])
def test_warmup_checks_adapter_before_synthesis(monkeypatch, ready, busy):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps({'ready': ready, 'busy': busy}).encode()
    opener = MagicMock()
    opener.open.return_value = response
    calls = []

    def build_opener(handler):
        calls.append(handler.proxies)
        return opener

    monkeypatch.setattr('alter_ego.voice.gpt_sovits.urllib.request.build_opener', build_opener)
    client = RemoteGPTSoVITS('http://127.0.0.1:9881/tts')
    if ready and not busy:
        asyncio.run(client.check_ready())
    else:
        with pytest.raises(RuntimeError, match='tts_not_ready'):
            asyncio.run(client.check_ready())
    assert calls == [{}]
    opener.open.assert_called_once_with('http://127.0.0.1:9881/health', timeout=5)
