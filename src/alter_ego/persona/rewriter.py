"""口吻改写器：本项目的核心自研模块。

分工：云端 LLM 负责「说什么」（内容生成），本模块负责「怎么说」（人格赋予）。
这不只是架构选择 —— 赛事规则将「仅调用通用模型接口」列为不认定创新，
本模块是「角色人格由自研模块生成」这一论证的实物支撑。

四层人格架构中本模块实现 ①②③④ 的③和④：
  ① 人格卡        → card.py
  ② few-shot 池   → examples.py
  ③ 本地口吻改写  → 本文件，跑在板子 CPU 上
  ④ 一致性自评    → 本文件 _score()，偏离则重生成

实测依据见 docs/实验数据.md。关键结论：
  · prefill「情绪：」把格式遵守率从 70% 提到 100%
  · 固定前缀 + 可变后缀使 prompt 处理耗时降 92%
  · 规则置于示例之后、原句之前，治住编造信息与人称翻转
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..contracts import EMOTIONS, Emotion, PersonaContext, RewriteResult
from .card import PersonaCard
from .examples import ExamplePool

_log = logging.getLogger(__name__)

SYSTEM = "你是一个严格按格式输出的改写工具。"

# 小模型会输出枚举外的近义词（实测 Sarashina 常给 cheerful / thoughtful）。
# 加一层别名映射比反复加强 prompt 约束更可靠。
_EMOTION_ALIAS: dict[str, str] = {
    "cheerful": "happy", "cheerfully": "happy", "glad": "happy",
    "joy": "happy", "joyful": "happy", "excited": "happy", "pleased": "happy",
    "fun": "happy",
    # sad / angry 已不在枚举（素材里没有哭也没有怒）。
    # 这些词全部收敛到 unhappy —— 枚举外的词必须映到**存在的立绘**上，
    # 否则 sprite_for() 会兜底成 neutral，表情完全丢失且不报错。
    "sad": "unhappy", "angry": "unhappy", "mad": "unhappy",
    "tired": "unhappy", "sleepy": "unhappy", "worried": "unhappy",
    "lonely": "unhappy", "down": "unhappy", "disappointed": "unhappy",
    "annoyed": "unhappy", "upset": "unhappy", "frustrated": "unhappy",
    "displeased": "unhappy", "sorry": "unhappy",
    "thoughtful": "neutral", "calm": "neutral", "normal": "neutral",
    "serious": "neutral", "explaining": "neutral",
    "curious": "surprised", "confused": "surprised", "puzzled": "surprised",
    "shocked": "surprised", "amazed": "surprised",
    "embarrassed": "shy", "bashful": "shy", "shyly": "shy", "flustered": "shy",
    "confident": "assertive", "firm": "assertive", "insistent": "assertive",
    "determined": "assertive", "emphatic": "assertive",
}

# 停止序列：v1 实测有 2/20 条停不下来，持续续写「原句：/改写：」刷满 token 上限
STOP = ["\n原句", "原句：", "<|im_end|>", "\n\n"]


@dataclass
class _Parsed:
    emotion: Emotion | None
    text: str | None


class LocalRewriter:
    """实现 contracts.PersonaRewriter。

    前缀在 __init__ 时构建一次并缓存 —— 它必须对同一角色逐字节恒定，
    否则 llama.cpp 的 KV cache 复用失效。
    """

    def __init__(
        self,
        card: PersonaCard,
        pool: ExamplePool,
        provider,                   # llm.providers.OpenAICompatProvider
        n_examples: int = 3,
    ):
        self.card = card
        self.pool = pool
        self.provider = provider
        self.last_error: str = ""      # 最近一次推理失败原因，供诊断
        self._prefix = self._build_prefix(n_examples)
        self._emotion_prefix = self._build_emotion_prefix()

        # 日文模式走规则引擎而非模型生成。依据见 docs/实验数据.md：
        # 模型方案三轮调优最高仅 35% 可用率（混英文、抄示例、语法崩坏），
        # 规则引擎 34/34 全对且快 150 倍。
        self._ja_style = None
        if card.language == "ja":
            from .ja_politeness import EndingStyle
            self._ja_style = EndingStyle.from_card(
                card.sentence_endings, card.fillers
            )

    # ---------------- prompt 构建 ----------------

    def _build_prefix(self, n_examples: int) -> str:
        """固定前缀。整段对同一角色恒定，是前缀缓存的命中对象。"""
        return (
            f"把句子改写成「{self.card.name}」的说话方式。\n\n"
            f"{self.card.name}的说话特点：\n{self.card.style_prompt()}\n\n"
            f"示例：\n{self.pool.render(n_examples)}\n\n"
            f"改写规则（务必遵守）：\n{self.card.rules_prompt()}\n\n"
            # 枚举值必须显式列出且靠近结尾。v2 曾删掉此段，
            # 情绪标签有效率从 70% 掉到 50%，解析失败全部兜底成 neutral，
            # 导致情绪立绘切换功能瘫掉一半。
            "情绪只能填这六个英文词之一，不要用中文，不要用其他词：\n"
            f"{' / '.join(EMOTIONS)}\n\n"
            "原句："
        )

    def _build_prompt(self, content: str, ctx: PersonaContext) -> str:
        """拼 ChatML 并以「情绪：」prefill。

        可变部分全部追加在最末尾，前缀区一个字节都不动。
        """
        tail = content
        hints: list[str] = []
        if ctx.user_name:
            hints.append(f"对话对象：{ctx.user_name}")
        if ctx.visual_hint:
            hints.append(f"当前场景：{ctx.visual_hint}")
        if hints:
            tail = f"{content}\n（{'；'.join(hints)}）"

        return (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{self._prefix}{tail}<|im_end|>\n"
            f"<|im_start|>assistant\n情绪："
        )

    # ---------------- 解析与自评 ----------------

    @staticmethod
    def _parse(raw: str) -> _Parsed:
        """解析 prefill 后的输出，形如 `happy\\n改写：今天天气超好的呢！`"""
        emotion: Emotion | None = None
        text: str | None = None
        lines = raw.strip().splitlines()

        if lines and lines[0].strip():
            head = lines[0].strip().strip("：:").split()[0]
            if head in EMOTIONS:
                emotion = head  # type: ignore[assignment]

        m = re.search(r"改写[：:]\s*(.+)", raw)
        if m:
            text = m.group(1).strip()
        elif len(lines) > 1:
            for ln in lines[1:]:
                if ln.strip():
                    text = ln.strip()
                    break
        return _Parsed(emotion, text)

    def _score(self, original: str, rewritten: str) -> tuple[bool, str]:
        """第④层一致性自评。规则打分，不调模型 —— 省一次推理。

        只检查能机械判定的硬缺陷，风格好坏不在此判断。
        """
        if not rewritten:
            return False, "空输出"
        if len(rewritten) > len(original) * 3 + 20:
            return False, "输出过长，疑似编造内容"
        # 语气词置于句首（v1 实测 2/20，如「呢？现在是晚上十点了啦！」）
        for w in self.card.sentence_endings:
            if rewritten.startswith(w):
                return False, f"语气词「{w}」置于句首"
        for bad in self.card.forbidden:
            if bad and bad in rewritten:
                return False, f"出现禁止内容「{bad}」"
        return True, ""

    # ---------------- 对外接口 ----------------

    def _build_emotion_prefix(self) -> str:
        """情绪分类用的固定前缀。

        与改写前缀分开，因为两者的任务不同：分类只需看句子判断情绪，
        不需要人格卡的语尾规则与 few-shot 示例。前缀更短 → 缓存更小、推理更快。
        """
        # few-shot 示例是必需的，不是装饰。
        # 初版无示例的 91 字前缀实测 5/8 准确（把「どうしてここに」判成 angry、
        # 「もう帰りましょう」判成 shy）。分类任务同样需要示范边界。
        return (
            "文を読んで、話し手の感情を一語で答えてください。\n"
            f"選択肢（これ以外は禁止）：{' / '.join(EMOTIONS)}\n\n"
            "例：\n"
            "文：今日はいい天気ですね。\n感情：happy\n\n"
            "文：少し疲れました。\n感情：sad\n\n"
            "文：どうしてここにいるのですか？\n感情：surprised\n\n"
            "文：今は夜の十時です。\n感情：neutral\n\n"
            "文：ほめてくれてありがとう。\n感情：shy\n\n"
            "文：もう我慢できません。\n感情：angry\n\n"
            "文："
        )

    def classify_emotion(self, text: str) -> Emotion:
        """只做情绪分类。输出 3 token 左右，远快于整句改写。

        Sarashina-1B 实测 20/20 准确 —— 分类任务比生成任务容易得多。
        失败兜底 neutral。
        """
        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{self._emotion_prefix}{text}<|im_end|>\n"
            f"<|im_start|>assistant\n感情："
        )
        try:
            raw = self.provider.completion(
                prompt, temperature=0.1, n_predict=6,
                stop=["\n", "<|im_end|>"],
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            _log.warning("情绪分类失败，兜底 neutral: %s", self.last_error)
            return "neutral"

        head = raw.strip().strip("：:").split()[0].lower() if raw.strip() else ""
        head = re.sub(r"[（(].*", "", head)
        if head in EMOTIONS:
            return head                              # type: ignore[return-value]
        return _EMOTION_ALIAS.get(head, "neutral")   # type: ignore[return-value]

    def rewrite_ja(
        self, content: str, ctx: PersonaContext, emotion: Emotion | None = None
    ) -> RewriteResult:
        """日文路径：规则引擎做语尾变换，不调模型。

        依据见 docs/实验数据.md —— 模型生成方案三轮调优最高 35% 可用率
        （混英文、抄 few-shot 示例、日语语法崩坏），规则引擎 34/34 全对且快 150 倍。

        emotion 由调用方传入（云端模型的结构化输出），为 None 时用关键词兜底。
        不在此处调小模型做分类：Sarashina-1B 实测仅 58%，且会额外花 0.22 s。
        """
        from .ja_politeness import guess_emotion, rewrite as ja_rewrite

        try:
            text, kind = ja_rewrite(content, self._ja_style)
        except Exception as exc:
            self.last_error = f"规则引擎异常 {type(exc).__name__}: {exc}"
            _log.warning(self.last_error)
            text, kind = content, "none"

        if kind == "none":
            # 规则未命中：输入可能已是常体，或含未覆盖的句式
            _log.debug("语尾规则未命中: %s", content)

        emo = emotion or guess_emotion(content)
        return RewriteResult(text=text, emotion=emo)   # type: ignore[arg-type]

    def rewrite(
        self, content: str, ctx: PersonaContext, emotion: Emotion | None = None
    ) -> RewriteResult:
        """改写并给出情绪标签。

        emotion 为云端模型给出的情绪，日文路径优先采用它。

        任何失败都返回原文兜底，绝不抛异常 ——
        宁可这句没改写，也不能让对话链路中断。
        """
        if self._ja_style is not None:
            return self.rewrite_ja(content, ctx, emotion)

        prompt = self._build_prompt(content, ctx)
        p = self.card.params
        attempts = 1 + max(0, p.retry_on_invalid)
        last_text = content
        last_emotion: Emotion = "neutral"

        for i in range(attempts):
            try:
                raw = self.provider.completion(
                    prompt,
                    temperature=p.temperature + i * 0.1,   # 重试时略增多样性
                    top_p=p.top_p,
                    n_predict=p.max_tokens,
                    stop=STOP,
                )
            except Exception as exc:
                # 推理失败走兜底，但必须留下痕迹 ——
                # 静默吞异常会让「改写不生效」看起来像「模型效果差」，
                # 实际可能是 server 没起、端口被占或代理拦截。
                self.last_error = f"{type(exc).__name__}: {exc}"
                _log.warning("改写推理失败，返回原文兜底: %s", self.last_error)
                break

            parsed = self._parse(raw)
            if parsed.emotion:
                last_emotion = parsed.emotion
            if parsed.text:
                last_text = parsed.text
                ok, _ = self._score(content, parsed.text)
                if ok:
                    return RewriteResult(text=parsed.text, emotion=last_emotion)

        return RewriteResult(text=last_text, emotion=last_emotion)

    def warmup(self) -> float:
        """用假数据跑一次，把前缀缓存填好。返回耗时秒数。

        冷启动首次改写实测 0.52-1.38 s，板子上推算 3 秒以上。
        必须在服务启动时预热，不能让它发生在用户第一次说话时。
        """
        import time
        t0 = time.perf_counter()
        self.rewrite("你好。", PersonaContext())
        return time.perf_counter() - t0
