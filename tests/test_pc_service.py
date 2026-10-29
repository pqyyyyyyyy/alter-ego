"""The PC preview must report real provider failures and resolve local art."""
import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
import requests
import uvicorn

from alter_ego.llm.providers import ProviderChainError
from pc.service import PcRuntime, create_app


class FakeRegistry:
    def __init__(self):
        self.fail = False
        self.messages = []

    def chat_with_fallback(self, messages, **kwargs):
        self.messages.append(messages)
        if self.fail:
            raise ProviderChainError("provider#1: HTTP401 authentication")
        return json.dumps({"text": "你好，今天过得怎么样？", "emotion": "happy"}), ""


@contextmanager
def running(app):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(.01)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive()


def request(base, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(urllib.request.Request(
            base + path, data=body,
            headers={"Content-Type": "application/json"} if body else {}), timeout=3) as response:
        return response.status, json.load(response)


def test_pc_real_reply_and_provider_failure_do_not_look_alike():
    registry = FakeRegistry()
    app = create_app(registry=registry)
    with running(app) as base:
        _, status = request(base, "/api/status")
        assert status["character"] == "小夜"
        assert status["sprite_url"] is None
        assert status["ready"]
        _, reply = request(base, "/api/chat", {"text": "你好"})
        assert reply["reply"] == "你好，今天过得怎么样？"
        assert reply["history_turns"] == 1
        assert registry.messages[0][-1]["content"] == "你好"

        registry.fail = True
        try:
            request(base, "/api/chat", {"text": "再说一句"})
            assert False, "a failed cloud call must not appear successful"
        except urllib.error.HTTPError as error:
            assert error.code == 502
            assert "鉴权失败" in json.load(error)["detail"]
        assert app.state.runtime.dialog.history_turns == 1
        _, reset = request(base, "/api/reset", {})
        assert reset["history_turns"] == 0


def test_pc_missing_key_does_not_call_cloud(monkeypatch):
    for key in ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "ZHIPU_API_KEY",
                "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    app = create_app()
    with running(app) as base:
        _, status = request(base, "/api/status")
        assert not status["ready"]
        try:
            request(base, "/api/chat", {"text": "你好"})
            assert False
        except urllib.error.HTTPError as error:
            assert error.code == 503
            assert "DASHSCOPE_API_KEY" in json.load(error)["detail"]


def test_pc_accepts_webp_only_sprite(tmp_path):
    (tmp_path / "persona.yaml").write_text(
        "name: Example\nemotions:\n  neutral: neutral.png\n", encoding="utf-8"
    )
    sprite_dir = tmp_path / "sprites" / "aligned"
    sprite_dir.mkdir(parents=True)
    (sprite_dir / "neutral.webp").write_bytes(b"webp")
    runtime = PcRuntime(tmp_path, "dashscope", False, FakeRegistry())
    assert runtime.sprite("neutral") == sprite_dir / "neutral.webp"
    assert runtime.sprite_url("neutral") == "/api/sprite/neutral"


@pytest.mark.parametrize("provider,key_env,base_url,model", [
    ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions",
     "deepseek-flash"),
    ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1/chat/completions",
     "gpt-6-luna"),
])
def test_pc_routes_selected_provider_with_its_own_key(
        monkeypatch, provider, key_env, base_url, model):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv(key_env, "test-key")
    app = create_app(provider_name=provider)
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "text": "真实后端测试回复", "emotion": "neutral"})}}]}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    app.state.runtime.registry.get(provider)._session.post = fake_post
    with running(app) as base:
        _, status = request(base, "/api/status")
        assert status["provider"] == provider
        assert status["model"] == model
        assert status["ready"]
        _, answer = request(base, "/api/chat", {"text": "你好"})
        assert answer["reply"] == "真实后端测试回复"
    assert len(calls) == 1
    assert calls[0][0] == base_url
    assert calls[0][1]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0][1]["json"]["model"] == model
    if provider == "openai":
        assert calls[0][1]["json"]["reasoning_effort"] == "none"
        assert "temperature" not in calls[0][1]["json"]


def test_pc_auto_uses_only_configured_key(monkeypatch):
    for key in ("DASHSCOPE_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    app = create_app()
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "text": "你好", "emotion": "neutral"})}}]}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    app.state.runtime.registry.get("deepseek")._session.post = fake_post
    with running(app) as base:
        _, status = request(base, "/api/status")
        assert status["provider"] == "deepseek"
        assert status["model"] == "deepseek-flash"
        _, answer = request(base, "/api/chat", {"text": "你好"})
        assert answer["provider"] == "deepseek"
    assert len(calls) == 1
    assert calls[0][1]["headers"]["Authorization"] == "Bearer deepseek-test-key"


def test_pc_auto_falls_back_to_next_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "invalid-test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    for key in ("ZHIPU_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    app = create_app()
    calls = []

    def fail_auth(url, **kwargs):
        calls.append("dashscope")
        response = requests.Response()
        response.status_code = 401
        raise requests.HTTPError(response=response)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "text": "备用平台回复", "emotion": "neutral"})}}]}

    def succeed(url, **kwargs):
        calls.append("deepseek")
        return Response()

    app.state.runtime.registry.get("dashscope")._session.post = fail_auth
    app.state.runtime.registry.get("deepseek")._session.post = succeed
    with running(app) as base:
        _, status = request(base, "/api/status")
        assert status["provider"] == "dashscope, deepseek"
        _, answer = request(base, "/api/chat", {"text": "你好"})
        assert answer["reply"] == "备用平台回复"
        assert answer["provider"] == "deepseek"
    assert calls == ["dashscope", "deepseek"]
