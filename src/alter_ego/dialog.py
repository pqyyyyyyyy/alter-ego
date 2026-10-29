"""对话编排层：把云端主模型、本地改写器、感知上下文串成一轮对话。

**这是此前整条链路缺失的一环。** 在它之前，`card.system_prompt()` 零调用处、
`providers.chat_with_fallback()` 从未被调用 —— 云端从来没被接进来过，
唯一能跑的入口 `/debug/say` 是跳过云端直接喂改写器的调试通道。

链路：

    用户中文（ASR 或 /debug/chat）
        ↓
    云端 chat()  system_prompt = 人格 + 台词参照 + 输出规格
        ↓  response_format={"type":"json_object"}
    {"ja": "会えてうれしいです。", "zh": "见到你很高兴。", "emotion": "happy"}
        ↓  只把 ja 送规则引擎
    LocalRewriter.rewrite_ja()   敬体 → 常体 + 角色语尾（4 ms，确定性 100%）
        ↓
    DialogTurn{speak_ja="会えてうれしいよ。", subtitle_zh, emotion}

**为什么云端出敬体而不直接出角色口吻**：赛事规则第三条第4款把「仅调用通用
模型接口」列为不认定创新，创新性与开源融合合计 40 分压在这条上。
云端负责「说什么」，本地自研规则引擎负责「怎么说」—— 这一步是该论证的实物支撑。
顺带的好处是换角色只改人格卡的语尾列表，云端 prompt 一个字不动。

**独立成模块而非塞进 server/app.py**：JSON 解析是静默失败高发区，
必须能脱离服务器单测。
"""
from __future__ import annotations

import json
import logging
import re
import time

from .persona.output_gate import japanese_error, SAFE_JA
from .contracts import EMOTIONS, CloudReply, DialogTurn, Emotion, PersonaContext
from .llm.providers import ProviderChainError, ProviderRegistry
from .persona.card import PersonaCard
from .persona.rewriter import _EMOTION_ALIAS, LocalRewriter

_log = logging.getLogger(__name__)

# 代码围栏。模型常自发把 JSON 包进 ```json ... ```，即使 prompt 里说了不要。
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# 退路：从一段文字里抠出最外层的 {...}
_BRACES = re.compile(r"\{.*\}", re.DOTALL)


def parse_cloud_reply(raw: str, language: str = "ja") -> tuple[CloudReply, str]:
    """Strict structure/type gate, then conservative language gate.
    Valid Japanese plain text is allowed as explicit subtitle-missing degradation.
    JSON arrays/scalars are never treated as spoken prose.
    """
    if not isinstance(raw,str):return CloudReply(ja=""), "云端输出类型错误"
    text=raw.strip()
    if not text:return CloudReply(ja="",raw=raw), "云端返回空内容"
    if len(text)>16384:return CloudReply(ja=""), "云端原始内容超长"
    data=None; parsed=False;note=""
    try:
        data=json.loads(text);parsed=True
    except (json.JSONDecodeError,TypeError):
        m=_FENCE.search(text) or _BRACES.search(text)
        if m:
            candidate=m.group(1) if m.re is _FENCE else m.group(0)
            try:
                data=json.loads(candidate);parsed=True;note="云端输出带代码围栏或夹杂说明，已剥离"
            except (json.JSONDecodeError,TypeError):pass
    if parsed and not isinstance(data,dict):
        return CloudReply(ja="",raw=raw), "JSON不是对象"
    if not parsed:
        error=japanese_error(text) if language=="ja" else ("too_long" if len(text)>2000 else "")
        if error:return CloudReply(ja="",raw=raw), "正文门禁:"+error
        return CloudReply(ja=text,zh="",raw=raw), "云端未按 JSON 输出，已整段当作正文（字幕缺失）"
    key="ja" if language=="ja" else "text"
    body=data.get(key,data.get("text" if language=="ja" else "zh",""))
    zh=data.get("zh","") if language=="ja" else body
    if body is None:body=""
    if zh is None:zh=""
    emo,emo_note=normalize_emotion(data.get("emotion"))
    notes=[n for n in (note,emo_note) if n]
    if not isinstance(body,str) or not isinstance(zh,str):
        return CloudReply(ja="",raw=raw,emotion=emo), "JSON字段类型错误"
    body=body.strip();zh=zh.strip()
    if not body:
        return CloudReply(ja="",zh=zh,raw=raw,emotion=emo), "JSON 里没有 "+key+" 字段"
    error=japanese_error(body) if language=="ja" else ("too_long" if len(body)>2000 else "")
    if error:
        return CloudReply(ja="",zh="",raw=raw,emotion=emo), "正文门禁:"+error
    if len(zh)>2000:return CloudReply(ja="",raw=raw,emotion=emo), "中文字幕超长"
    if any(0xD800<=ord(c)<=0xDFFF for c in zh):
        return CloudReply(ja="",raw=raw,emotion=emo), "中文字幕Unicode错误"
    if language=="ja" and not zh:notes.append("云端未给 zh，字幕缺失")
    return CloudReply(ja=body,zh=zh,emotion=emo,raw=raw), "；".join(notes)


def _looks_like_bad_request(exc: BaseException) -> bool:
    """Only retry format-parameter incompatibility, not auth/input/network errors."""
    msg = str(exc).lower()
    return "response_format" in msg and any(word in msg for word in
        ("unsupported", "not support", "invalid", "unknown", "not allowed", "400", "422"))


def _cloud_failure_note(exc: BaseException) -> str:
    # Registry summaries are locally sanitized; arbitrary exception text is not.
    detail = str(exc) if isinstance(exc, ProviderChainError) else type(exc).__name__
    return '云端全部失败: ' + detail


def normalize_emotion(value: object) -> tuple[Emotion, str]:
    """情绪校验。枚举外的词走别名映射，仍不认识则 neutral。

    复用 rewriter._EMOTION_ALIAS 而不另写一份 —— 两份映射会在改枚举时漂移，
    10-05 改情绪枚举时就踩过这个：三处硬编码的 EMOTIONS 各自不一致。
    """
    if not isinstance(value, str) or not value.strip():
        return "neutral", "云端未给 emotion，兜底 neutral"
    key = value.strip().lower()
    if key in EMOTIONS:
        return key, ""                                  # type: ignore[return-value]
    mapped = _EMOTION_ALIAS.get(key)
    if mapped:
        return mapped, f"情绪「{value}」经别名映射为 {mapped}"  # type: ignore[return-value]
    return "neutral", f"情绪「{value}」不在枚举内，兜底 neutral"


class DialogOrchestrator:
    """一轮对话的编排。线程不安全；调用方必须用P01串行，旧服务未保证。"""

    def __init__(
        self,
        card: PersonaCard,
        rewriter: LocalRewriter,
        registry: ProviderRegistry,
        max_turns: int = 6,
        cloud_budget_s: float = 30.0,
    ) -> None:
        self.card = card
        self.rewriter = rewriter
        self.registry = registry
        self.max_turns = max_turns
        if cloud_budget_s <= 0: raise ValueError("positive budget required")
        self.cloud_budget_s = cloud_budget_s
        self.last_error: str = ""

        # system prompt 构建一次并缓存。它必须逐字节恒定，
        # 否则云端的 prompt 缓存失效（本地实测前缀缓存省 92% prompt 处理耗时，
        # 云端同理且直接体现在计费上）。
        self._system = card.system_prompt()

        # 多轮历史，OpenAI messages 格式。
        # **不用 PersonaContext.recent_turns** —— 它是 list[str]，没有角色标记，
        # 表达不了多轮；而这个格式是云端原生接受的形式。
        self._history: list[dict[str, str]] = []

    # ---------------- 对外接口 ----------------

    def reply(self, user_text: str, ctx: PersonaContext | None = None, event_hook=None) -> DialogTurn:
        """跑完一轮。任何失败都返回可用的 DialogTurn，绝不抛异常。"""
        ctx = ctx or PersonaContext()
        turn = DialogTurn(user_text=user_text, speak_ja="")
        notes: list[str] = []

        emit = event_hook or (lambda event: None)
        emit("cloud_start")
        try:
            reply, cloud_note, cloud_sec = self._ask_cloud(user_text, ctx)
        finally:
            emit("cloud_end")
        turn.cloud_sec = round(cloud_sec, 3)
        if cloud_note:
            notes.append(cloud_note)

        if not reply.ja:
            # 云端什么也没给出来，用人格卡的预设回复。
            # 这里**不调规则引擎** —— 预设回复本来就该写成角色口吻。
            preset = self.card.profile.fallback_reply
            if self.card.language == "ja" and japanese_error(preset):
                preset = SAFE_JA
                notes.append("预设回复未通过日文门禁")
            turn.speak_ja = preset
            turn.subtitle_zh = ""
            turn.emotion = "neutral"
            turn.error = "；".join(notes) or "云端无有效输出"
            self.last_error = turn.error
            _log.warning("对话降级为预设回复: %s", turn.error)
            return turn

        t0 = time.perf_counter()
        emit("rewrite_start")
        try:
            result = self.rewriter.rewrite(reply.ja, ctx, emotion=reply.emotion)
        finally:
            emit("rewrite_end")
        turn.rewrite_sec = round(time.perf_counter() - t0, 4)
        if self.rewriter.last_error:
            notes.append(f"改写: {self.rewriter.last_error}")

        turn.speak_ja = result.text
        if self.card.language == "ja" and japanese_error(turn.speak_ja):
            turn.speak_ja = SAFE_JA
            notes.append("改写结果未通过日文门禁")
            turn.subtitle_zh = ""
            turn.emotion = "neutral"
            turn.error = "；".join(notes)
            self.last_error = turn.error
            return turn
        turn.subtitle_zh = reply.zh
        turn.emotion = result.emotion
        turn.error = "；".join(notes)
        self.last_error = turn.error

        self._remember(user_text, reply)
        return turn

    def reset(self) -> None:
        """清空多轮历史。换用户时调用 —— 不然新用户会继承上一个人的对话。"""
        self._history.clear()

    @property
    def history_turns(self) -> int:
        return len(self._history) // 2

    # ---------------- 内部 ----------------

    def _ask_cloud(
        self, user_text: str, ctx: PersonaContext
    ) -> tuple[CloudReply, str, float]:
        messages = [{"role": "system", "content": self._system}]
        messages += self._history
        messages.append({"role": "user", "content": self._user_message(user_text, ctx)})

        t0 = time.perf_counter()
        try:
            raw, chain_note = self.registry.chat_with_fallback(
                messages,
                preset="",                 # 不在这层用 preset，交给 reply() 统一处理
                response_format={"type": "json_object"},
                temperature=0.8,
            )
        except Exception as exc:
            # 可能是 response_format 不被支持（并非所有 OpenAI 兼容实现都认它），
            # 去掉它重试 —— prompt 里已写了格式要求，解析器也能剥围栏。
            #
            # 但**只在看起来是参数问题时才重试**。网络故障下重试会白等第二轮
            # 全链超时（30 s × 链长 × 2），演示现场卡两分钟比降级更糟。
            if (not _looks_like_bad_request(exc) or
                time.perf_counter() - t0 >= self.cloud_budget_s):
                return (
                    CloudReply(ja=""),
                    _cloud_failure_note(exc),
                    time.perf_counter() - t0,
                )
            _log.info("疑似不支持 response_format，去掉它重试: %s", exc)
            try:
                raw, chain_note = self.registry.chat_with_fallback(
                    messages, preset="", temperature=0.8,
                )
                chain_note = f"response_format 不被支持已去掉；{chain_note}".strip("；")
            except Exception as exc2:
                return (
                    CloudReply(ja=""),
                    _cloud_failure_note(exc2),
                    time.perf_counter() - t0,
                )
        cost = time.perf_counter() - t0

        if cost > self.cloud_budget_s:
            return CloudReply(ja=""), "云端整体预算超限，迟到内容丢弃", cost
        reply, parse_note = parse_cloud_reply(raw, self.card.language)
        note = "；".join(n for n in (chain_note, parse_note) if n)
        return reply, note, cost

    def _user_message(self, user_text: str, ctx: PersonaContext) -> str:
        """拼 user message。

        感知上下文放这里而不是 system —— system 必须恒定以保住 prompt 缓存。
        """
        hints: list[str] = []
        if ctx.user_name:
            hints.append(f"说话的人是{ctx.user_name}")
        if ctx.visual_hint:
            hints.append(ctx.visual_hint)
        if hints:
            return f"{user_text}\n（{'；'.join(hints)}）"
        return user_text

    def _remember(self, user_text: str, reply: CloudReply) -> None:
        """记入历史。

        存的是云端原始的**敬体** ja，不是改写后的常体 —— 让云端看到自己
        之前说的话（它只认自己的输出格式），而不是规则引擎的产物。
        混进常体会让它以为可以直接输出常体，破坏敬体约定。
        """
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": reply.ja})
        keep = self.max_turns * 2
        if len(self._history) > keep:
            del self._history[: len(self._history) - keep]
