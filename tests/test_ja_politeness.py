"""日语敬体转常体与角色语尾的回归测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.persona.ja_politeness import EndingStyle, rewrite  # noqa: E402


def test_かも语尾不会在人格卡解析时丢失():
    style = EndingStyle.from_card(["かも"], [])

    assert style.declarative == ["かも"]
    text, kind = rewrite("分かりました。", style)

    assert kind == "masu"
    assert text == "分かったかも。"


def test_かもね也作为陈述句语尾保留():
    style = EndingStyle.from_card(["かもね"], [])

    assert style.declarative == ["かもね"]
