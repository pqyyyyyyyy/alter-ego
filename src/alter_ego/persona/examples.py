"""few-shot 示例池：角色语料，改写效果的主要来源。

⚠️ 一个必须遵守的架构约束 —— 示例选择必须是**确定性**的。

llama.cpp 的前缀缓存命中条件是 prompt 前缀逐字节相同。实测数据：
缓存命中时 prompt 处理从 298 token / 562 ms 降到 11.5 token / 44 ms（省 92%）。
板子上这个差距是 4-5 秒 vs 0.3 秒，直接决定产品可用性。

因此示例池**不能按输入动态检索** —— 每次注入不同示例会让前缀变化，
缓存全部失效。这是性能与「相关性更高的示例」之间的取舍，
在边缘设备上性能优先。

动态检索的收益（示例更贴合输入）远小于缓存失效的代价，
除非将来改成「按情境分桶 + 每桶独立缓存槽」，但那需要
llama-server 的多 slot 管理，不在本期范围。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    situation: str
    plain: str       # 中性说法，像机器说的
    styled: str      # 角色会怎么说


class ExamplePool:
    def __init__(self, items: list[Example]):
        self._items = items

    @classmethod
    def load(cls, path: str | Path) -> ExamplePool:
        items: list[Example] = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    d = json.loads(line)
                    items.append(Example(
                        situation=d.get("situation", ""),
                        plain=d["plain"],
                        styled=d["styled"],
                    ))
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"{path}:{lineno} 格式错误（需含 plain 与 styled 字段）: {exc}"
                    ) from exc
        return cls(items)

    def __len__(self) -> int:
        return len(self._items)

    def situations(self) -> list[str]:
        """已覆盖的情境，用于检查语料是否均衡。"""
        seen: dict[str, int] = {}
        for it in self._items:
            seen[it.situation] = seen.get(it.situation, 0) + 1
        return [f"{k}×{v}" for k, v in sorted(seen.items())]

    def select(self, k: int = 3) -> list[Example]:
        """选出注入 prompt 的示例。**确定性**选择，不看输入内容。

        策略：每个情境取第一条，尽量覆盖不同句型。
        v1 实测用 5 条同质示例出现了「直接抄示例」的泄漏（3/20），
        减到 3 条且句型分散后消失。
        """
        picked: list[Example] = []
        used_situations: set[str] = set()
        for it in self._items:
            if it.situation not in used_situations:
                picked.append(it)
                used_situations.add(it.situation)
            if len(picked) >= k:
                break
        # 情境种类不足时用剩余条目补齐
        if len(picked) < k:
            for it in self._items:
                if it not in picked:
                    picked.append(it)
                if len(picked) >= k:
                    break
        return picked

    def render(self, k: int = 3) -> str:
        """渲染成 prompt 片段。输出对同一池子恒定，保住前缀缓存。"""
        return "\n\n".join(
            f"原句：{e.plain}\n情绪：{_guess_emotion(e)}\n改写：{e.styled}"
            for e in self.select(k)
        )


def _guess_emotion(ex: Example) -> str:
    """示例若未标情绪，按情境关键词粗判。

    仅用于渲染 few-shot 示例 —— 示例里展示情绪字段是为了让模型学会
    输出这个字段，标得不够准不影响实际改写。
    真实情绪由模型在推理时自行判定。
    """
    s = ex.situation
    # 取值必须在 contracts.EMOTIONS 内。2026-10-05 枚举按素材重定后
    # sad/angry 已移除，这里原先映射到 sad 的两项改为 unhappy。
    table = {
        "开心": "happy", "被夸": "shy", "打招呼": "happy",
        "失落": "unhappy", "累": "unhappy", "疑惑": "surprised",
        "惊讶": "surprised", "关心": "neutral", "不知道": "neutral",
        "主张": "assertive", "纠正": "assertive",
    }
    for key, val in table.items():
        if key in s:
            return val
    return "neutral"
