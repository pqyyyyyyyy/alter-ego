"""0.5B 本地模型的**日文**口吻改写能力验证。

决策实验：Qwen2.5-0.5B 是中文为主的模型，日文能力明显弱。
本脚本测它能否胜任日文的语尾改写（キャラ語尾）。

日语其实比中文更适合这套方案 —— 「〜かな」「〜だよ」这类角色语尾
辨识度远高于中文的「呢啦哦」，模型要学的映射关系更清晰。
但前提是模型本身的日文能力够用。

用法：
    llama-server.exe -m <0.5B模型> -c 2048 -t 4 --port 8089 --cache-reuse 256
    python scripts/bench_rewrite_ja.py

判定线（人工看「合格+勉强」）：
    ≥70% 继续用 0.5B
    50-70% 换 Qwen2.5-1.5B 或 Qwen3-0.6B
    <50% 该模型规模的日文能力不足，必须换模型
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

import requests

ENDPOINT = "http://127.0.0.1:8089/completion"

# 枚举与别名从 contracts / rewriter 导入，不在这里复制一份。
# 原先两个基准脚本各自硬编码了一份 EMOTIONS，枚举按素材重定（2026-10-05）
# 后立刻漂移成了三份不一致的定义 —— 基准脚本用旧枚举评分，
# 结果会无声地判错，而指标看起来完全正常。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alter_ego.contracts import EMOTIONS                      # noqa: E402
from alter_ego.persona.rewriter import _EMOTION_ALIAS as _ALIAS  # noqa: E402

SYSTEM = "あなたは指定された形式で厳密に出力する日本語の書き換えツールです。"

# 固定前缀。结构与中文版一致：示例在前，规则在后（靠近原句）。
# 语尾用七海千秋的标志性「〜かな」「〜だよ」「〜ね」。
PREFIX = """次の文を「千秋」の話し方に書き換えてください。

千秋の話し方：
- のんびりしていて、落ち着いている
- 語尾に かな / だよ / ね / よ をよく使う
- 迷うときは うーん / えっと で始める

例：
原文：今日はいい天気です。
感情：happy
書き換え：今日はいい天気だね。

原文：少し疲れました。
感情：sad
書き換え：うーん、ちょっと疲れちゃったかな。

原文：どうしてここにいるのですか？
感情：surprised
書き換え：えっと、どうしてここにいるのかな？

書き換えのルール（必ず守ること）：
1. 話し方だけを変え、意味は変えない
2.「あなた」と「わたし」を入れ替えない
3. 原文にない情報を追加しない
4. 語尾は文末だけに置き、文頭に置かない
5. 上の例の内容をそのまま使わない
6. 英語を混ぜない。すべて日本語で書く

出力は必ず二行。一行目は感情、二行目は書き換え。
感情は次の六語のどれか一つだけ（これ以外の語は禁止）：
neutral / happy / sad / surprised / angry / shy

例：
感情：happy
書き換え：今日はいい天気だね。

原文："""

# 20 条测试句。覆盖陈述/疑问/否定/感情/长句，贴近真实对话。
CASES = [
    "今日は晴れです。",
    "今は時間がありません。",
    "今日は元気がなさそうですね。",
    "この問題は複雑で、説明に時間がかかります。",
    "会えてうれしいです。",
    "その本を取ってください。",
    "意味がよく分かりません。",
    "外は雨が降り始めました。",
    "三時間も続けて работа…休んだほうがいいです。",
    "このことはもう少し考えます。",
    "あなたの考えは面白いですね。",
    "すみません、聞き取れませんでした。",
    "前に話していたことを覚えています。",
    "今は夜の十時です。",
    "水を飲みませんか？",
    "結果は予想と違いました。",
    "ここで待っています。",
    "どうして急にそんなことを聞くのですか？",
    "それは危険かもしれません。",
    "今日はここまでにしましょう。",
]
# 修正第 9 条里混入的俄文词
CASES[8] = "三時間も続けて作業しています。少し休んだほうがいいです。"


def build_prompt(src: str) -> str:
    """ChatML + prefill「感情：」锁死输出格式。"""
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{PREFIX}{src}<|im_end|>\n"
        f"<|im_start|>assistant\n感情："
    )


def ask(src: str, timeout: int = 60) -> tuple[str, float, float]:
    payload = {
        "prompt": build_prompt(src),
        "temperature": 0.6,
        "top_p": 0.9,
        "n_predict": 64,
        # 不含 "\n\n" —— Sarashina-1B 在感情行后会输出空行，
        # 带上它会把输出截断在第一行，实测导致 9/20 解析失败
        "stop": ["\n原文", "原文：", "<|im_end|>"],
        "stream": True,
        "cache_prompt": True,
    }
    t0 = time.perf_counter()
    ttft = -1.0
    chunks: list[str] = []
    with requests.post(ENDPOINT, json=payload, stream=True, timeout=timeout,
                       proxies={"http": None, "https": None}) as r:
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
    emotion = text = None
    lines = raw.strip().splitlines()
    if lines and lines[0].strip():
        head = lines[0].strip().strip("：:").split()[0].lower()
        head = re.sub(r"[（(].*", "", head)          # 去掉 "tired（疲労）" 的括注
        if head in EMOTIONS:
            emotion = head
        else:
            emotion = _ALIAS.get(head)              # cheerful -> happy 之类
    m = re.search(r"書き換え[：:]\s*(.+)", raw)
    if m:
        text = m.group(1).strip()
    elif len(lines) > 1:
        for ln in lines[1:]:
            if ln.strip():
                text = ln.strip()
                break
    return emotion, text


# 日文语尾检查用：出现这些才算带上了角色语尾
ENDINGS = ("かな", "だよ", "ね", "よ", "かしら")
FILLERS = ("うーん", "えっと", "あの")


def main() -> None:
    print(f"端点 {ENDPOINT}")
    print(f"固定前缀 {len(PREFIX)} 字符，测试 {len(CASES)} 条（日文）\n")
    print("=" * 80)

    ttfts: list[float] = []
    totals: list[float] = []
    ok_format = ok_emotion = has_ending = 0
    failures: list[tuple[str, str]] = []

    for i, src in enumerate(CASES, 1):
        try:
            raw, ttft, total = ask(src)
        except Exception as exc:
            print(f"[{i:2d}] 请求失败: {type(exc).__name__}: {exc}")
            failures.append((src, f"请求异常 {type(exc).__name__}"))
            continue

        emotion, text = parse(raw)
        ttfts.append(ttft)
        totals.append(total)

        if text:
            ok_format += 1
            if any(text.rstrip("。！？!?").endswith(e) for e in ENDINGS) \
               or any(e in text for e in ENDINGS):
                has_ending += 1
        else:
            failures.append((src, f"解析失败: {raw[:60]!r}"))
        if emotion:
            ok_emotion += 1

        print(f"[{i:2d}] {ttft:5.2f}s首字 {total:5.2f}s总  {emotion or '??':9s} {src}")
        print(f"     → {text or '（解析失败）'}")

    n = len(ttfts)
    if not n:
        print("\n全部失败，检查 llama-server 是否在 8089 端口")
        return

    N = len(CASES)
    print("=" * 80)
    print(f"格式遵守率    {ok_format}/{N}  ({ok_format / N:.0%})")
    print(f"情绪标签有效  {ok_emotion}/{N}  ({ok_emotion / N:.0%})")
    print(f"带角色语尾    {has_ending}/{N}  ({has_ending / N:.0%})  ← 日文改写的关键指标")
    print(f"首字延迟      均值 {statistics.mean(ttfts):.2f}s  中位 {statistics.median(ttfts):.2f}s")
    print(f"总耗时        均值 {statistics.mean(totals):.2f}s  中位 {statistics.median(totals):.2f}s")

    if failures:
        print(f"\n失败 {len(failures)} 条：")
        for src, why in failures:
            print(f"  · {src}  —  {why}")

    print("\n【人工判断】每条的 → 行：日文是否自然、语尾是否像千秋、意思有没有变。")
    print("按「合格+勉强」计数。≥70% 用 0.5B；50-70% 换 1.5B；<50% 日文能力不足。")


if __name__ == "__main__":
    main()
