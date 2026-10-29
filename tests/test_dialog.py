"""对话编排与云端响应解析的测试。

JSON 解析是这条链路的静默失败高发区：格式不对不会报错，
只会让字幕消失、情绪全倒向 neutral、日文变成未改写的敬体。
每一条降级路径都必须有测试，且必须断言 `error` 字段非空 ——
**容错和可观测性是两件事**，能跑不等于跑对了。

    python -m pytest tests/test_dialog.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.contracts import EMOTIONS, PersonaContext      # noqa: E402
from alter_ego.dialog import (                                # noqa: E402
    DialogOrchestrator,
    normalize_emotion,
    parse_cloud_reply,
)
from alter_ego.persona.card import PersonaCard                # noqa: E402
from alter_ego.persona.examples import ExamplePool            # noqa: E402
from alter_ego.persona.rewriter import LocalRewriter          # noqa: E402

CARD_JA = {
    "name": "千秋",
    "language": "ja",
    "style": {
        "tone": "のんびり",
        "sentence_endings": ["ね", "よ", "かな"],
        "fillers": ["うーん"],
    },
    "profile": {
        "identity": "ゲームが好きな友達。",
        "speech_samples": ["ゲームしようよ。", "……そうかな。"],
        "fallback_reply": "うーん…今はちょっと考えられないかな。",
    },
    "emotions": {e: [f"{e}"] for e in EMOTIONS},
}


# ---------------- 测试替身 ----------------

class FakeRegistry:
    """按脚本返回云端响应。raise_with 非 None 时抛异常。"""

    def __init__(self, replies: list[str] | None = None,
                 raise_with: BaseException | None = None) -> None:
        self.replies = replies or []
        self.raise_with = raise_with
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    def chat_with_fallback(self, messages, preset="", chain=None, **kw):
        self.calls.append(messages)
        self.kwargs.append(kw)
        if self.raise_with is not None:
            raise self.raise_with
        i = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[i], ""


def make_card(tmp_path: Path, data: dict) -> PersonaCard:
    p = tmp_path / "persona.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return PersonaCard.load(p)


def make_orch(tmp_path: Path, registry, data: dict | None = None,
              max_turns: int = 6) -> DialogOrchestrator:
    card = make_card(tmp_path, data or CARD_JA)
    pool = ExamplePool([])
    rewriter = LocalRewriter(card, pool, None)
    return DialogOrchestrator(card, rewriter, registry, max_turns=max_turns)


# ---------------- 解析器 ----------------

def test_正常json():
    r, note = parse_cloud_reply(
        '{"ja":"会えてうれしいです。","zh":"见到你很高兴。","emotion":"happy"}'
    )
    assert r.ja == "会えてうれしいです。"
    assert r.zh == "见到你很高兴。"
    assert r.emotion == "happy"
    assert note == ""


def test_带代码围栏():
    """模型常自发加 ```json，即使 prompt 说了不要。"""
    raw = '```json\n{"ja":"はい。","zh":"是的。","emotion":"neutral"}\n```'
    r, note = parse_cloud_reply(raw)
    assert r.ja == "はい。"
    assert "围栏" in note


def test_json前后夹杂说明文字():
    raw = '好的，这是我的回答：\n{"ja":"はい。","zh":"是的。","emotion":"neutral"}\n希望有帮助'
    r, note = parse_cloud_reply(raw)
    assert r.ja == "はい。"
    assert note


def test_缺zh字段时字幕为空且告警():
    r, note = parse_cloud_reply('{"ja":"はい。","emotion":"happy"}')
    assert r.ja == "はい。"
    assert r.zh == ""
    assert "字幕缺失" in note


def test_缺emotion字段兜底neutral():
    r, note = parse_cloud_reply('{"ja":"はい。","zh":"是的。"}')
    assert r.emotion == "neutral"
    assert "emotion" in note


def test_枚举外情绪走别名映射():
    """复用 rewriter 的别名表。两份映射会在改枚举时漂移。"""
    r, note = parse_cloud_reply('{"ja":"はい。","zh":"b","emotion":"sad"}')
    assert r.emotion == "unhappy", "sad 已不在枚举，应映射到 unhappy"
    assert "别名" in note


def test_完全不认识的情绪兜底neutral():
    r, note = parse_cloud_reply('{"ja":"はい。","zh":"b","emotion":"螺旋升天"}')
    assert r.emotion == "neutral"
    assert "不在枚举" in note


def test_完全不是json则整段当正文():
    """模型即使不守格式，输出的通常仍是一句可用的日语。"""
    r, note = parse_cloud_reply("会えてうれしいです。")
    assert r.ja == "会えてうれしいです。"
    assert r.zh == ""
    assert "未按 JSON" in note


def test_空响应():
    r, note = parse_cloud_reply("")
    assert r.ja == ""
    assert "空内容" in note


def test_json里没有ja字段():
    r, note = parse_cloud_reply('{"zh":"只有中文","emotion":"happy"}')
    assert r.ja == ""
    assert "没有 ja" in note


def test_中文路径读text字段():
    r, note = parse_cloud_reply(
        '{"text":"今天天气不错。","emotion":"happy"}', language="zh"
    )
    assert r.ja == "今天天气不错。"
    assert r.zh == "今天天气不错。"
    assert note == ""


def test_原始响应始终保留():
    """解析失败时要能看到云端到底返回了什么。"""
    raw = "乱七八糟"
    r, _ = parse_cloud_reply(raw)
    assert r.raw == raw


@pytest.mark.parametrize("value", [None, "", 123, [], {}])
def test_情绪非字符串都兜底(value):
    emo, note = normalize_emotion(value)
    assert emo == "neutral"
    assert note


def test_情绪大小写不敏感():
    assert normalize_emotion("HAPPY")[0] == "happy"


# ---------------- 编排 ----------------

def test_完整一轮(tmp_path):
    reg = FakeRegistry(['{"ja":"分かりました。","zh":"我明白了。","emotion":"happy"}'])
    orch = make_orch(tmp_path, reg)
    turn = orch.reply("你明白了吗")

    assert turn.user_text == "你明白了吗"
    assert turn.subtitle_zh == "我明白了。"
    assert turn.emotion == "happy"
    assert turn.error == ""
    # 规则引擎把敬体转成了常体 + 角色语尾
    assert turn.speak_ja != "分かりました。"
    assert "ました" not in turn.speak_ja
    assert turn.speak_ja.startswith("分かった")


def test_规则引擎收到的是ja字段而非中文(tmp_path):
    """链路的关键断言。喂中文进规则引擎会产出「今天天气不错ね。」这类垃圾且不报错。"""
    reg = FakeRegistry(['{"ja":"疲れました。","zh":"我有点累了。","emotion":"unhappy"}'])
    orch = make_orch(tmp_path, reg)
    turn = orch.reply("你累吗")

    assert turn.speak_ja.startswith("疲れた"), "应是日文形态变换的结果"
    assert "累" not in turn.speak_ja, "中文不该进规则引擎"


def test_请求带了response_format(tmp_path):
    reg = FakeRegistry(['{"ja":"はい。","zh":"b","emotion":"happy"}'])
    orch = make_orch(tmp_path, reg)
    orch.reply("hi")
    assert reg.kwargs[0].get("response_format") == {"type": "json_object"}


def test_system_prompt只发一次且含关键要求(tmp_path):
    reg = FakeRegistry(['{"ja":"はい。","zh":"b","emotion":"happy"}'])
    orch = make_orch(tmp_path, reg)
    orch.reply("hi")

    msgs = reg.calls[0]
    assert msgs[0]["role"] == "system"
    sys_text = msgs[0]["content"]
    assert "です" in sys_text, "必须要求敬体，否则规则引擎接不住"
    assert "json" in sys_text.lower()
    assert "ゲームしようよ。" in sys_text, "台词参照应进 system prompt"


def test_感知上下文进user消息而非system(tmp_path):
    """system 必须逐字节恒定以保住 prompt 缓存。"""
    reg = FakeRegistry(['{"ja":"はい。","zh":"b","emotion":"happy"}'])
    orch = make_orch(tmp_path, reg)
    orch.reply("hi", PersonaContext(user_name="小明", visual_hint="画面里是小明"))

    msgs = reg.calls[0]
    assert "小明" not in msgs[0]["content"]
    assert "小明" in msgs[-1]["content"]
    assert "画面里是小明" in msgs[-1]["content"]


def test_system_prompt跨轮不变(tmp_path):
    reg = FakeRegistry(['{"ja":"はい。","zh":"b","emotion":"happy"}'])
    orch = make_orch(tmp_path, reg)
    orch.reply("一")
    orch.reply("二")
    assert reg.calls[0][0]["content"] == reg.calls[1][0]["content"]


# ---------------- 多轮历史 ----------------

def test_历史被带入下一轮(tmp_path):
    reg = FakeRegistry(['{"ja":"はい。","zh":"是的。","emotion":"neutral"}'])
    orch = make_orch(tmp_path, reg)
    orch.reply("第一句")
    orch.reply("第二句")

    msgs = reg.calls[1]
    contents = [m["content"] for m in msgs]
    assert "第一句" in contents
    assert "第二句" in contents[-1]


def test_历史存的是敬体原文而非改写结果(tmp_path):
    """让云端看到自己的输出格式。混进常体会让它以为可以直接输出常体。"""
    reg = FakeRegistry(['{"ja":"分かりました。","zh":"明白了。","emotion":"neutral"}'])
    orch = make_orch(tmp_path, reg)
    orch.reply("一")
    orch.reply("二")

    assistant = [m for m in reg.calls[1] if m["role"] == "assistant"]
    assert assistant[0]["content"] == "分かりました。"


def test_历史截断在max_turns(tmp_path):
    reg = FakeRegistry(['{"ja":"はい。","zh":"b","emotion":"neutral"}'])
    orch = make_orch(tmp_path, reg, max_turns=2)
    for i in range(5):
        orch.reply(f"第{i}句")
    assert orch.history_turns == 2
    contents = [m["content"] for m in reg.calls[-1]]
    assert "第0句" not in contents


def test_reset清空历史(tmp_path):
    reg = FakeRegistry(['{"ja":"はい。","zh":"b","emotion":"neutral"}'])
    orch = make_orch(tmp_path, reg)
    orch.reply("一")
    assert orch.history_turns == 1
    orch.reset()
    assert orch.history_turns == 0


def test_失败的轮次不进历史(tmp_path):
    """降级轮不该污染上下文。"""
    reg = FakeRegistry(["不是json也不是日语的空白"])
    orch = make_orch(tmp_path, reg)
    orch.reply("一")
    # 整段当正文的降级仍算有输出，会进历史；但云端全失败的不会
    reg2 = FakeRegistry(raise_with=RuntimeError("全部 provider 失败: timeout"))
    orch2 = make_orch(tmp_path, reg2)
    orch2.reply("一")
    assert orch2.history_turns == 0


# ---------------- 降级 ----------------

def test_云端全失败走预设回复(tmp_path):
    reg = FakeRegistry(raise_with=RuntimeError("全部 provider 失败: ConnectTimeout"))
    orch = make_orch(tmp_path, reg)
    turn = orch.reply("你好")

    assert turn.speak_ja == "うーん…今はちょっと考えられないかな。"
    assert turn.error, "降级必须留下痕迹，否则看起来和成功一样"
    assert orch.last_error


def test_网络故障不重试第二轮(tmp_path):
    """重试会白等第二轮全链超时，演示现场卡两分钟比降级更糟。"""
    reg = FakeRegistry(raise_with=RuntimeError("ConnectTimeout"))
    orch = make_orch(tmp_path, reg)
    orch.reply("你好")
    assert len(reg.calls) == 1


def test_参数错误才去掉response_format重试(tmp_path):
    class Flaky(FakeRegistry):
        def chat_with_fallback(self, messages, preset="", chain=None, **kw):
            self.calls.append(messages)
            self.kwargs.append(kw)
            if "response_format" in kw:
                raise RuntimeError("400 Bad Request: response_format unsupported")
            return '{"ja":"はい。","zh":"是的。","emotion":"neutral"}', ""

    reg = Flaky()
    orch = make_orch(tmp_path, reg)
    turn = orch.reply("你好")

    assert len(reg.calls) == 2
    assert "response_format" not in reg.kwargs[1]
    assert turn.subtitle_zh == "是的。"
    assert turn.error


def test_任何情况都不抛异常(tmp_path):
    for reg in (
        FakeRegistry(raise_with=RuntimeError("boom")),
        FakeRegistry([""]),
        FakeRegistry(["{}"]),
        FakeRegistry(['{"ja":null,"zh":null,"emotion":null}']),
    ):
        orch = make_orch(tmp_path, reg)
        turn = orch.reply("你好")
        assert isinstance(turn.speak_ja, str)
        assert turn.emotion in EMOTIONS


def test_耗时被记录(tmp_path):
    reg = FakeRegistry(['{"ja":"分かりました。","zh":"明白。","emotion":"neutral"}'])
    orch = make_orch(tmp_path, reg)
    turn = orch.reply("hi")
    assert turn.cloud_sec >= 0
    assert turn.rewrite_sec >= 0


# ---------------- 中文路径 ----------------

def test_中文路径输出规格不同(tmp_path):
    data = {**CARD_JA, "language": "zh",
            "profile": {**CARD_JA["profile"], "speech_samples": []}}
    card = make_card(tmp_path, data)
    spec = card.output_spec()
    assert '"text"' in spec
    assert '"ja"' not in spec


def test_日文路径缺台词会告警(tmp_path):
    data = {**CARD_JA, "profile": {"identity": "x", "fallback_reply": "y"}}
    card = make_card(tmp_path, data)
    assert any("speech_samples" in w for w in card.validate())


def test_缺预设回复会告警(tmp_path):
    data = {**CARD_JA, "profile": {"identity": "x", "speech_samples": ["a"]}}
    card = make_card(tmp_path, data)
    assert any("fallback_reply" in w for w in card.validate())
