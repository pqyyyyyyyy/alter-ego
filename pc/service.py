"""A small PC chat service that leaves the board runtime untouched."""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from alter_ego.contracts import EMOTIONS, RewriteResult
from alter_ego.dialog import DialogOrchestrator
from alter_ego.llm.providers import ProviderChainError, ProviderRegistry
from alter_ego.persona.card import PersonaCard
from alter_ego.persona.examples import ExamplePool
from alter_ego.persona.rewriter import LocalRewriter


ROOT = Path(__file__).resolve().parents[1]
PC_DIR = Path(__file__).resolve().parent


class ChatRequest(BaseModel):
    text: str


class PassThroughRewriter:
    """PC text preview when the optional local rewrite model is absent."""

    last_error = ""

    def rewrite(self, content, context, emotion=None):
        return RewriteResult(text=content, emotion=emotion or "neutral")


class SelectedRegistry:
    def __init__(self, registry: ProviderRegistry, provider_names: list[str]):
        self.registry = registry
        self.provider_names = provider_names
        self.active_name = ""
        self.last_failures: list[tuple[str, str]] = []

    def chat_with_fallback(self, messages, **kwargs):
        self.active_name = ""
        self.last_failures = []
        for name in self.provider_names:
            options = kwargs.copy()
            if name == "openai":
                options.pop("temperature", None)
                options["reasoning_effort"] = "none"
            try:
                reply, note = self.registry.chat_with_fallback(
                    messages, chain=[name], **options
                )
            except ProviderChainError as exc:
                reason = re.search(r"HTTP\d{3} [a-z_]+|request_timeout|tls_error|"
                                   r"proxy_error|connection_error|response_format_error|"
                                   r"provider_error", str(exc))
                self.last_failures.append((name, reason.group(0) if reason else "调用失败"))
                continue
            self.active_name = name
            return reply, note
        raise ProviderChainError("; ".join(f"{name}: {reason}"
                                            for name, reason in self.last_failures))


class PcRuntime:
    def __init__(self, character_dir: Path, provider_name: str, local_rewrite: bool,
                 registry=None):
        self.character_dir = character_dir.resolve()
        self.card = PersonaCard.load(self.character_dir / "persona.yaml")
        self.provider_name = provider_name
        self.registry = registry or ProviderRegistry(str(PC_DIR / "providers.yaml"))
        self.lock = asyncio.Lock()

        if registry is None:
            candidates = (self.registry.dialog_chain if provider_name == "auto"
                          else [provider_name])
            self.provider_names = [name for name in candidates
                                   if os.environ.get(self.registry.get(name).cfg.api_key_env)]
            self.missing_key = not self.provider_names
            self.key_env = ", ".join(self.registry.get(name).cfg.api_key_env
                                     for name in (self.provider_names or candidates))
            self.model = ", ".join(self.registry.get(name).cfg.model
                                   for name in self.provider_names)
            selected_registry = SelectedRegistry(self.registry, self.provider_names)
        else:
            self.provider_names = []
            self.missing_key = False
            self.key_env = ""
            self.model = ""
            selected_registry = registry
        self.selected_registry = selected_registry

        examples_path = self.character_dir / "examples.jsonl"
        pool = ExamplePool.load(examples_path) if examples_path.exists() else ExamplePool([])
        if self.card.language == "ja" or local_rewrite:
            rewrite_provider = self.registry.get("local") if local_rewrite else None
            rewriter = LocalRewriter(self.card, pool, rewrite_provider)
        else:
            rewriter = PassThroughRewriter()
        self.local_rewrite = local_rewrite
        self.dialog = DialogOrchestrator(self.card, rewriter, selected_registry)

    def sprite(self, emotion: str) -> Path | None:
        if emotion not in EMOTIONS:
            return None
        sprite_dir = (self.character_dir / "sprites" / "aligned").resolve()
        for name in self.card.sprites_for(emotion):
            if Path(name).name != name:
                continue
            path = Path(name)
            candidates = ([path.with_suffix(".webp"), path] if path.suffix.lower() == ".png"
                          else [path])
            for candidate in candidates:
                resolved = (sprite_dir / candidate).resolve()
                if sprite_dir in resolved.parents and resolved.is_file():
                    return resolved
        return None

    def sprite_url(self, emotion: str) -> str | None:
        return f"/api/sprite/{emotion}" if self.sprite(emotion) else None


def create_app(character_dir: Path | None = None, provider_name: str = "auto",
               local_rewrite: bool = False, registry=None) -> FastAPI:
    runtime = PcRuntime(character_dir or ROOT / "characters" / "example",
                        provider_name, local_rewrite, registry)
    app = FastAPI(title="Alter-Ego PC")
    app.state.runtime = runtime
    app.mount("/static", StaticFiles(directory=str(PC_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(PC_DIR / "index.html")

    @app.get("/api/background")
    async def background():
        return FileResponse(ROOT / "web" / "background-green-fullscreen.png")

    @app.get("/api/status")
    async def status():
        return {
            "character": runtime.card.name,
            "language": runtime.card.language,
            "provider": ", ".join(runtime.provider_names) or runtime.provider_name,
            "model": runtime.model,
            "ready": not runtime.missing_key,
            "issue": f"请设置以下任一平台的 API Key 后重启服务：{runtime.key_env}。"
                     if runtime.missing_key else "",
            "local_rewrite": runtime.local_rewrite or runtime.card.language == "ja",
            "sprite_url": runtime.sprite_url("neutral"),
        }

    @app.get("/api/sprite/{emotion}")
    async def sprite(emotion: str):
        path = runtime.sprite(emotion)
        if path is None:
            raise HTTPException(404, "立绘不存在")
        return FileResponse(path)

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        message = request.text.strip()
        if not message or len(message) > 1000:
            raise HTTPException(400, "请输入 1 到 1000 字的消息。")
        if runtime.missing_key:
            raise HTTPException(503, f"请设置 {runtime.key_env} 后重启服务。")
        async with runtime.lock:
            previous_turns = runtime.dialog.history_turns
            result = await asyncio.to_thread(runtime.dialog.reply, message)
            if runtime.dialog.history_turns == previous_turns or not result.speak_ja.strip():
                error = result.error
                failures = getattr(runtime.selected_registry, "last_failures", [])
                if failures and not getattr(runtime.selected_registry, "active_name", ""):
                    detail = "；".join(f"{name}: {reason}" for name, reason in failures)
                elif "HTTP401" in error:
                    detail = "对话服务鉴权失败。"
                elif "HTTP402" in error:
                    detail = "对话服务余额不足。"
                elif "HTTP403" in error:
                    detail = "对话服务拒绝访问，请检查账号权限。"
                elif "HTTP404" in error:
                    detail = "对话模型或接口地址不存在，请检查 PC provider 配置。"
                elif "HTTP429" in error:
                    detail = "对话服务限流或额度不足。"
                else:
                    detail = "对话服务未能生成回答，请检查网络、模型和服务日志。"
                raise HTTPException(502, detail)
            return {
                "reply": result.speak_ja,
                "subtitle": result.subtitle_zh,
                "emotion": result.emotion,
                "sprite_url": runtime.sprite_url(result.emotion),
                "history_turns": runtime.dialog.history_turns,
                "provider": getattr(runtime.selected_registry, "active_name", ""),
                "model": (runtime.registry.get(runtime.selected_registry.active_name).cfg.model
                          if getattr(runtime.selected_registry, "active_name", "") else ""),
                "rewrite_note": "本地改写不可用，已显示原文。"
                if result.error.startswith("改写:") else "",
            }

    @app.post("/api/reset")
    async def reset():
        async with runtime.lock:
            runtime.dialog.reset()
        return {"ok": True, "history_turns": 0}

    return app
