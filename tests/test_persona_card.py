"""人格卡的立绘映射与状态动画测试。

这层逻辑错了不会报错，只会表现成「某个表情从来不出现」或「待机没动画」，
而那时候你会先去怀疑情绪判定或前端。所以映射解析必须有测试钉住。

    python -m pytest tests/test_persona_card.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.contracts import EMOTIONS                      # noqa: E402
from alter_ego.persona.card import PersonaCard                # noqa: E402

MINIMAL = {
    "name": "测试角色",
    "language": "ja",
    "style": {"tone": "のんびり", "sentence_endings": ["ね", "よ"]},
    # 日文角色必须有这两项，否则 validate 告警（见 test_dialog.py 的对应用例）：
    #   speech_samples —— 日文路径不走 examples.jsonl，语气参照只能放这里
    #   fallback_reply —— 云端全失败时的预设回复
    "profile": {
        "identity": "测试用",
        "speech_samples": ["ゲームしようよ。"],
        "fallback_reply": "うーん…",
    },
    "emotions": {
        "neutral": ["a", "b"],
        "happy": ["c"],
        "surprised": ["d"],
        "shy": ["e"],
        "unhappy": ["f"],
        "assertive": ["g"],
    },
    "animations": {
        "idle": {"frames": ["s1"], "loop": True},
        "wake": {"frames": ["s1", "s2", "s3"], "interval_ms": 1000},
    },
}


def write_card(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "persona.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture()
def card(tmp_path: Path) -> PersonaCard:
    return PersonaCard.load(write_card(tmp_path, MINIMAL))


# ---------------- 情绪映射 ----------------

def test_列表写法解析并自动补png(card):
    assert card.sprites_for("neutral") == ["a.png", "b.png"]
    assert card.sprites_for("happy") == ["c.png"]


def test_单字符串写法也兼容(tmp_path):
    d = {**MINIMAL, "emotions": {**MINIMAL["emotions"], "happy": "solo.png"}}
    c = PersonaCard.load(write_card(tmp_path, d))
    assert c.sprites_for("happy") == ["solo.png"]


def test_已带png后缀的不重复追加(tmp_path):
    d = {**MINIMAL, "emotions": {**MINIMAL["emotions"], "happy": ["x.png"]}}
    c = PersonaCard.load(write_card(tmp_path, d))
    assert c.sprites_for("happy") == ["x.png"]


def test_多候选图按index轮换(card):
    got = [card.sprite_for("neutral", i) for i in range(5)]
    assert got == ["a.png", "b.png", "a.png", "b.png", "a.png"]


def test_单候选图轮换始终同一张(card):
    assert {card.sprite_for("happy", i) for i in range(10)} == {"c.png"}


def test_index越界取模而不是兜底neutral(card):
    """越界返回 neutral 会让情绪静默丢失，必须取模。"""
    assert card.sprite_for("happy", 999) == "c.png"


def test_未映射的情绪兜底到neutral(card):
    assert card.sprite_for("不存在的情绪") == "a.png"


def test_连neutral都没有时不崩(tmp_path):
    d = {**MINIMAL, "emotions": {}}
    c = PersonaCard.load(write_card(tmp_path, d))
    assert c.sprite_for("happy") == "neutral.png"


# ---------------- 状态动画 ----------------

def test_动画帧解析(card):
    a = card.animation("wake")
    assert a is not None
    assert a.frames == ["s1.png", "s2.png", "s3.png"]
    assert a.interval_ms == 1000
    assert a.loop is False


def test_idle动画循环(card):
    a = card.animation("idle")
    assert a is not None and a.loop is True


def test_动画简写只给帧列表(tmp_path):
    d = {**MINIMAL, "animations": {"idle": ["only"]}}
    c = PersonaCard.load(write_card(tmp_path, d))
    a = c.animation("idle")
    assert a is not None and a.frames == ["only.png"]
    assert a.interval_ms == 1000       # 默认值


def test_不存在的动画返回None(card):
    assert card.animation("没有这个") is None


def test_动画序列化给前端的格式(card):
    d = card.animation("wake").to_dict()
    assert set(d) == {"frames", "interval_ms", "loop"}


def test_动画不占情绪枚举(card):
    """state 和 emotion 是两条独立消息，动画名不该混进情绪里。"""
    for key in card.animations:
        assert key not in EMOTIONS


# ---------------- validate ----------------

def test_缺情绪会告警(tmp_path):
    d = {**MINIMAL, "emotions": {"neutral": ["a"]}}
    c = PersonaCard.load(write_card(tmp_path, d))
    warns = c.validate()
    assert any("缺少情绪立绘映射" in w for w in warns)


def test_枚举外的情绪会告警(tmp_path):
    """写了云端永远不会输出的情绪，那些立绘是死的。"""
    d = {**MINIMAL, "emotions": {**MINIMAL["emotions"], "angry": ["old"]}}
    c = PersonaCard.load(write_card(tmp_path, d))
    warns = c.validate()
    assert any("枚举外的情绪" in w for w in warns)
    assert any("angry" in w for w in warns)


def test_完整卡片无告警(card):
    assert card.validate() == []


def test_立绘文件缺失会告警(card, tmp_path):
    """映射里写错文件名不报错，只会静默兜底成 neutral。必须查文件存在性。"""
    sprites = tmp_path / "sprites"
    sprites.mkdir()
    for n in ["a", "b", "c", "d", "e", "f"]:          # 故意漏掉 g 和 s1-s3
        (sprites / f"{n}.png").write_bytes(b"x")
    warns = card.validate(sprite_dir=sprites)
    msg = " ".join(warns)
    assert "立绘文件缺失" in msg
    assert "g.png" in msg
    assert "s1.png" in msg                            # 动画帧也要查
    assert "a.png" not in msg


def test_立绘目录不存在会告警(card, tmp_path):
    warns = card.validate(sprite_dir=tmp_path / "没有这个目录")
    assert any("立绘目录不存在" in w for w in warns)


def test_文件齐全时无缺失告警(card, tmp_path):
    sprites = tmp_path / "sprites"
    sprites.mkdir()
    for n in ["a", "b", "c", "d", "e", "f", "g", "s1", "s2", "s3"]:
        (sprites / f"{n}.png").write_bytes(b"x")
    assert card.validate(sprite_dir=sprites) == []


# ---------------- 真实角色卡 ----------------

def test_nanami卡片与实际立绘一致():
    """这条测试会随素材变动而失败，那正是它的用途 ——
    改了 persona.yaml 的映射或动了 sprites/aligned/ 就该知道。
    """
    card_path = ROOT / "characters" / "nanami" / "persona.yaml"
    sprites = ROOT / "characters" / "nanami" / "sprites" / "aligned"
    if not card_path.exists() or not sprites.is_dir():
        pytest.skip("nanami 角色包不在（已 gitignore，CI 上没有）")
    c = PersonaCard.load(card_path)
    assert c.language == "ja", "日文规则引擎靠这个字段启用"
    assert c.validate(sprite_dir=sprites) == []
    assert c.animation("dialog").frames == ["wondering-1.png"]
