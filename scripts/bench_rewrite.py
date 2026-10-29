"""0.5B 本地模型的口吻改写能力验证。

这是项目的决策实验：验证 0.5B 规模的模型能否胜任"把中性句改写成角色口吻"。
结论决定核心创新点（本地人格层）是否成立。

用法：
    1. 先起 llama-server：
       llama-server.exe -m <0.5B模型> -c 2048 -t 4 --port 8089 --cache-reuse 256
    2. python scripts/bench_rewrite.py

产出：逐条改写结果 + 汇总指标，复制进 docs/实验数据.md
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

import requests

# 走 /completion 裸接口而非 /v1/chat/completions：
# 需要自己拼 ChatML 模板，换来的是能用 prefill 锁死输出格式 ——
# prompt 以「情绪：」结尾，模型只能从这里续写，无法自由发挥开头。
ENDPOINT = "http://127.0.0.1:8089/completion"

# 枚举从 contracts 导入，不在这里复制。见 bench_rewrite_ja.py 同处的说明 ——
# 重复定义的枚举会在改动后无声漂移，让基准脚本用旧枚举判错而指标看着正常。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alter_ego.contracts import EMOTIONS                      # noqa: E402

# ===================== 固定前缀 =====================
# 设计要点（v2，依据 v1 实测失败模式修正）：
#   1. 示例减到 3 条且句型分散 —— v1 用 5 条，出现了直接抄示例的泄漏
#   2. 改写规则移到示例之后、原句之前 —— 小模型对靠近结尾的指令敏感得多
#   3. 显式禁止人称互换、添加信息、语气词置于句首 —— 均为 v1 实测缺陷
#   4. 可变内容（原句）仍在最末尾，保住前缀缓存（实测省 92% prompt 耗时）
SYSTEM = "你是一个严格按格式输出的中文改写工具。"

PREFIX = """把句子改写成「小夜」的说话方式。

小夜的说话特点：
- 活泼，略带撒娇
- 句尾爱用 呢 / 啦 / 哦 / 吧
- 犹豫时开头说 唔 / 欸

示例：
原句：今天天气不错。
情绪：happy
改写：今天天气超好的呢！

原句：我有点累了。
情绪：sad
改写：唔…我有点累了哦。

原句：你怎么会在这里？
情绪：surprised
改写：欸？你怎么会在这里啦？

改写规则（务必遵守）：
1. 只改说话方式，不改意思
2.「你」和「我」不能互换
3. 不添加原句里没有的信息
4. 语气词只放句尾，不能放句首
5. 不要照搬上面示例的内容

情绪只能填这六个英文词之一，不要用中文，不要用其他词：
neutral / happy / sad / surprised / angry / shy

原句："""

# ===================== 20 条测试句 =====================
# 覆盖陈述/疑问/否定/情感/长句等不同类型，尽量贴近真实对话
CASES = [
    "今天是晴天。",
    "我现在没有时间。",
    "你今天看起来心情不太好。",
    "这个问题比较复杂，需要慢慢解释。",
    "我很高兴见到你。",
    "请把那本书递给我。",
    "我不太明白你的意思。",
    "外面开始下雨了。",
    "你已经连续工作三个小时了，应该休息一下。",
    "这件事我需要再考虑一下。",
    "你的想法很有意思。",
    "抱歉，我刚才没听清。",
    "我记得你上次提到过这件事。",
    "现在是晚上十点了。",
    "你要不要喝点水？",
    "这个结果和我预期的不一样。",
    "我在这里等你回来。",
    "你为什么突然这么问？",
    "那样做可能会有风险。",
    "今天就先到这里吧。",
]


def build_prompt(src: str) -> str:
    """拼 ChatML 模板，并以「情绪：」做 prefill 锁死输出格式。

    关键是最后那个 "情绪：" —— 它是助手回复的开头，模型只能从此续写，
    不能再自由发挥。v1 的 6 条格式失败全部由此消除。
    """
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{PREFIX}{src}<|im_end|>\n"
        f"<|im_start|>assistant\n情绪："
    )


def ask(src: str, timeout: int = 60) -> tuple[str, float, float]:
    """发一次改写请求（流式），返回 (原始输出, 首字延迟秒, 总耗时秒)。

    流式是为了单独测首字延迟 —— 它反映 prompt 处理开销，
    是前缀缓存有没有生效的温度计。
    """
    payload = {
        "prompt": build_prompt(src),
        "temperature": 0.6,          # v1 用 0.7，偏高导致编造；降一点更稳
        "top_p": 0.9,
        "n_predict": 48,             # v1 实测平均输出 26 token，48 够用
        "stop": ["\n原句", "原句：", "<|im_end|>", "\n\n"],
        "stream": True,
        "cache_prompt": True,        # 显式开启前缀缓存
    }
    t0 = time.perf_counter()
    ttft = -1.0
    chunks: list[str] = []

    with requests.post(ENDPOINT, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            body = raw[6:].strip()
            if body == b"[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            piece = obj.get("content") or ""
            if piece:
                if ttft < 0:
                    ttft = time.perf_counter() - t0
                chunks.append(piece)
            if obj.get("stop"):
                break

    return "".join(chunks), max(ttft, 0.0), time.perf_counter() - t0


def parse(raw: str) -> tuple[str | None, str | None]:
    """解析 prefill 后的输出。

    因为 prompt 以「情绪：」结尾，模型的输出形如：
        happy\n改写：今天天气超好的呢！
    所以第一行直接就是情绪值，不需要再匹配「情绪：」前缀。
    """
    emotion = text = None
    lines = raw.strip().splitlines()

    if lines:
        head = lines[0].strip().strip("：:").split()[0] if lines[0].strip() else ""
        if head in EMOTIONS:
            emotion = head

    m = re.search(r"改写[：:]\s*(.+)", raw)
    if m:
        text = m.group(1).strip()
    elif len(lines) > 1:
        # 退路：模型漏了「改写：」标签，取第二行非空内容
        for ln in lines[1:]:
            if ln.strip():
                text = ln.strip()
                break
    return emotion, text


def main() -> None:
    print(f"端点 {ENDPOINT}")
    print(f"固定前缀长度 {len(PREFIX)} 字符，测试 {len(CASES)} 条\n")
    print("=" * 78)

    ttfts: list[float] = []
    totals: list[float] = []
    ok_format = 0          # 格式解析成功
    ok_emotion = 0         # 情绪标签有效
    failures: list[tuple[str, str]] = []

    for i, src in enumerate(CASES, 1):
        try:
            raw, ttft, total = ask(src)
        except Exception as exc:
            print(f"[{i:2d}] 请求失败: {exc}")
            failures.append((src, f"请求异常 {exc}"))
            continue

        emotion, text = parse(raw)
        ttfts.append(ttft)
        totals.append(total)

        if text:
            ok_format += 1
        else:
            failures.append((src, f"格式解析失败，原始输出: {raw[:60]!r}"))
        if emotion:
            ok_emotion += 1

        print(f"[{i:2d}] {ttft:5.2f}s首字 {total:5.2f}s总  "
              f"{emotion or '??':9s} {src}")
        print(f"     → {text or '（解析失败）'}")

    n = len(ttfts)
    if not n:
        print("\n全部请求失败，检查 server 是否在 8089 端口运行")
        return

    print("=" * 78)
    print(f"格式遵守率   {ok_format}/{len(CASES)}  ({ok_format / len(CASES):.0%})")
    print(f"情绪标签有效 {ok_emotion}/{len(CASES)}  ({ok_emotion / len(CASES):.0%})")
    print(f"首字延迟     均值 {statistics.mean(ttfts):.2f}s  "
          f"中位 {statistics.median(ttfts):.2f}s  "
          f"首条 {ttfts[0]:.2f}s  末条 {ttfts[-1]:.2f}s")
    print(f"总耗时       均值 {statistics.mean(totals):.2f}s  "
          f"中位 {statistics.median(totals):.2f}s")

    drop = ttfts[0] - statistics.mean(ttfts[1:]) if n > 1 else 0.0
    print(f"\n前缀缓存效果：首条比后续平均慢 {drop:.2f}s", end="  ")
    print("（明显为正 → 缓存生效）" if drop > 0.05 else "（差异不明显，检查 --cache-reuse）")

    if failures:
        print(f"\n失败 {len(failures)} 条：")
        for src, why in failures:
            print(f"  · {src}  —  {why}")

    print("\n改写质量需人工判断：上面每条的 → 行是否像那个口吻。")
    print("把成功条数记进 docs/实验数据.md。")


if __name__ == "__main__":
    main()
