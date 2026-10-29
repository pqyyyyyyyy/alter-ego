"""模块间接口契约。

改动此文件必须三人同步 —— 它是并行开发的唯一共享依赖。

分工（2026-10-03 调整）：
  perception/ persona/ llm/ server/ scheduler/  → 队长
  voice/                                        → z
  web/                                          → w
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal, Protocol

# numpy 只在类型注解里出现，而本文件有 `from __future__ import annotations`，
# 注解会被延迟求值成字符串，运行时不解析。放进 TYPE_CHECKING 后：
#   · 人格层（只读 YAML）、语音层（只用 sherpa-onnx）不必安装 numpy
#   · 类型检查器（mypy/pyright）照常能校验
# 感知层实际用到 ndarray 时自行 import numpy 即可。
if TYPE_CHECKING:
    import numpy as np

# ======================== 全局枚举 ========================
# 前后端共用，前端按 Emotion 取立绘文件名

State = Literal["IDLE", "NOTICED", "DIALOG", "SPEAKING"]

# 情绪枚举（2026-10-05 按实际立绘素材重定）。
#
# **枚举由素材与人设共同决定，不是拍一组通用情绪就完事。** 两处改动：
#   · 去掉 `angry` —— 七海是慵懒温柔的观察者，人设里不发怒，素材里也没有。
#     留着的后果是云端判出 angry 后演出不符合人设的表现。换成 `unhappy`（不悦）。
#   · 去掉 `sad` —— 这套素材里没有哭/失落的立绘，强行拿别的顶会名不副实。
#   · 加入 `assertive` —— 主张/强调。学级裁判里的七海是会据理力争的，
#     而且素材有两级强度（指/推手掌）。
#
# 数量仍是 6。这是刻意的上限：情绪由**云端模型从枚举里挑**，
# 选项越多挑得越噪（实测 Sarashina-1B 在 6 项上已有 58% 的误判倒向 neutral）。
# 立绘的丰富度靠「一个情绪映射多张候选图、本地轮换」实现，不靠扩枚举。
Emotion = Literal[
    "neutral", "happy", "surprised", "shy", "unhappy", "assertive",
]

EMOTIONS: tuple[str, ...] = (
    "neutral", "happy", "surprised", "shy", "unhappy", "assertive",
)

EMBEDDING_DIM = 512          # 人脸特征维度，ArcFace 系列固定 512
AUDIO_SAMPLE_RATE = 16000    # ASR 输入采样率，不可改
WS_PROTOCOL_VERSION = "aeg-ws-v2"


# ======================== 感知层（队长实现）========================

@dataclass
class FaceBox:
    """人脸检测框，像素坐标。

    `kps` 是 5 点关键点（左眼、右眼、鼻尖、左嘴角、右嘴角），shape (5, 2) float32。
    它不是可选的装饰 —— ArcFace 系列必须靠这 5 点做仿射对齐才能提特征，
    只有框没有关键点就提不了特征。留 None 默认值是为了让纯检测降级方案
    （只判断有人/没人）不必伪造数据。
    """
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray | None = None


@dataclass
class Identity:
    """身份比对结果。user_id 为 None 表示未注册访客。"""
    user_id: str | None
    name: str | None
    similarity: float        # 0~1，余弦相似度


class PerceptionBackend(Protocol):
    """推理后端。ONNX 版与昇腾版实现同一接口，可热替换。

    这个抽象是为了 10-05 上板时只换实现、不改业务逻辑。
    """

    def detect_faces(self, frame: np.ndarray) -> list[FaceBox]:
        """输入 BGR 图像，返回人脸框列表。"""
        ...

    def extract_embedding(self, frame: np.ndarray, box: FaceBox) -> np.ndarray:
        """提取人脸特征。

        返回 shape (512,) 的 float32，已做 L2 归一化。
        实现内部必须先用 `box.kps` 做仿射对齐再送入模型 ——
        漏掉对齐不会报错，但识别率会崩塌（实测同一人相似度从 0.6 掉到 0.2）。
        `box.kps` 为 None 时应抛异常，不要静默返回一个错的向量。
        """
        ...


class IdentityStore(Protocol):
    """用户档案库。只存特征向量，不存原始人脸图像。"""

    def enroll(self, name: str, embeddings: list[np.ndarray]) -> str:
        """注册新用户，返回 user_id。需用户显式触发。"""
        ...

    def identify(self, embedding: np.ndarray) -> Identity:
        ...

    def delete(self, user_id: str) -> None:
        """删除用户全部数据，包括磁盘文件。"""
        ...

    def list_users(self) -> list[tuple[str, str]]:
        """返回 [(user_id, name), ...]。"""
        ...


# ======================== 语音层（z 实现）========================

class ASREngine(Protocol):
    """流式语音识别。"""

    def feed(self, pcm_s16le_16k: bytes) -> None:
        """喂入音频片段。必须是 16kHz 单声道 16bit PCM。"""
        ...

    def poll(self) -> tuple[str, bool] | None:
        """返回 (文本, 是否最终结果)；无结果时返回 None。"""
        ...

    def reset(self) -> None:
        """一句话结束后必须调用，否则下一句会接在上一句后面。"""
        ...


class TTSEngine(Protocol):
    def synthesize(self, text: str) -> tuple[bytes, int]:
        """返回 (wav 字节, 采样率)。"""
        ...


class AsyncWavTTS(Protocol):
    async def synthesize_wav(self, text: str) -> bytes:
        """Complete only when the underlying request has actually finished."""
        ...


class AsyncMP3TTS(Protocol):
    async def synthesize_mp3(self, text: str) -> bytes:
        ...


# ======================== 人格层（队长实现）========================

@dataclass
class PersonaContext:
    """改写时的上下文。"""
    user_name: str | None = None
    visual_hint: str | None = None           # 视觉上下文摘要
    recent_turns: list[str] = field(default_factory=list)


@dataclass
class RewriteResult:
    text: str
    emotion: Emotion


@dataclass
class CloudReply:
    """云端主对话模型的结构化输出。

    **`ja` 必须是敬体（です・ます）。** 规则引擎是敬体→常体的形态变换器，
    给它常体或中文都会静默失败 —— 门禁正则一条不命中，但末尾仍会拼上
    日文终助词，产出「今天天气不错ね。」这类垃圾且不报错。
    """
    ja: str                      # 日文敬体，喂给规则引擎
    zh: str = ""                 # 中文字幕，是 ja 的翻译（保证语义严格对应）
    emotion: Emotion = "neutral"
    raw: str = ""                # 原始响应，解析失败时留作诊断


@dataclass
class DialogTurn:
    """一轮完整对话的产出。

    字幕不放进 RewriteResult —— 那是改写器的输出，而改写器只做日文形态变换，
    字幕来自云端。混在一起会让职责模糊。
    """
    user_text: str               # ASR 识别的中文
    speak_ja: str                # 规则引擎改写后的日文（常体 + 角色语尾），送 TTS
    subtitle_zh: str = ""        # 中文字幕
    emotion: Emotion = "neutral"
    cloud_sec: float = 0.0       # 云端耗时，诊断用
    rewrite_sec: float = 0.0
    error: str = ""              # 非空表示走了降级路径。容错必须配可观测性


class PersonaRewriter(Protocol):
    """把中性内容改写成角色口吻，并给出情绪标签。

    本项目核心模块：云端 LLM 负责"说什么"，此模块负责"怎么说"。
    解析失败时必须返回原文兜底，不得抛异常中断对话链路。
    """

    def rewrite(self, content: str, ctx: PersonaContext) -> RewriteResult:
        ...


# ======================== LLM 层（队长实现）========================

class LLMProvider(Protocol):
    """OpenAI 兼容接口。云端 API 与本地 llama.cpp 共用同一抽象。"""

    def chat(self, messages: list[dict[str, str]], **kw) -> str:
        ...

    def stream(self, messages: list[dict[str, str]], **kw) -> Iterator[str]:
        ...
