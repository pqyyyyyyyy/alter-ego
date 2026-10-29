"""The public character must remain usable without private assets or API access."""
import asyncio
import importlib

from alter_ego.dialog import DialogOrchestrator
from alter_ego.llm.providers import ProviderChainError
from alter_ego.persona.card import PersonaCard
from alter_ego.persona.examples import ExamplePool
from alter_ego.persona.rewriter import LocalRewriter


def test_public_character_has_visible_reply_when_cloud_fails():
    module = importlib.import_module("alter_ego.server.app")
    card = PersonaCard.load(module.CHARACTER / "persona.yaml")

    class UnavailableRegistry:
        def chat_with_fallback(self, messages, **kwargs):
            raise ProviderChainError("authentication failed")

    dialog = DialogOrchestrator(card, LocalRewriter(card, ExamplePool([]), None),
                                UnavailableRegistry())
    reply = dialog.reply("你好")
    assert reply.speak_ja == card.profile.fallback_reply
    assert reply.speak_ja.strip()
    assert "authentication failed" in reply.error


def test_missing_sprites_are_not_advertised_or_sent(tmp_path, monkeypatch):
    module = importlib.import_module("alter_ego.server.app")
    runtime = module.Runtime()
    runtime.card = PersonaCard(name="Example", tone="", emotions={
        "neutral": ["missing.png", "present.png"], "happy": ["missing.png"]})
    runtime.char_dir = tmp_path
    sprite_dir = tmp_path / "sprites" / "aligned"
    sprite_dir.mkdir(parents=True)
    (sprite_dir / "present.png").write_bytes(b"image")
    sent = []

    async def capture(message):
        sent.append(message)

    runtime.hub.send = capture
    monkeypatch.setattr(module, "rt", runtime)
    manifest = asyncio.run(module.debug_sprites())
    assert manifest["emotions"]["neutral"] == ["present.png"]
    assert manifest["emotions"]["happy"] == []
    assert asyncio.run(runtime.send_emotion("happy")) is None
    assert sent[-1] == {"type": "emotion", "value": "happy"}
    assert asyncio.run(runtime.send_emotion("neutral")) == "present.png"
    assert sent[-1]["sprite"] == "present.png"
