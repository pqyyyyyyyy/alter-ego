"""模块间接口契约。

改动此文件必须三人同步 —— 它是并行开发的唯一共享依赖。

分工：
  perception/ voice/   → w
  web/                 → z
  persona/ llm/ server/ scheduler/ → 队长
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Protocol

import numpy as np

# ======================== 全局枚举 ========================
# 前后端共用，前端按 Emotion 取立绘文件名

State = Literal["IDLE", "NOTICED", "DIALOG", "SPEAKING"]
Emotion = Literal["neutral", "happy", "sad", "surprised", "angry", "shy"]

EMOTIONS: tuple[str, ...] = (
    "neutral", "happy", "sad", "surprised", "angry", "shy",
)

EMBEDDING_DIM = 512          # 人脸特征维度，ArcFace 系列固定 512
AUDIO_SAMPLE_RATE = 16000    # ASR 输入采样率，不可改


# ======================== 感知层（w 实现）========================

@dataclass
class FaceBox:
    """人脸检测框，像素坐标。"""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


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
        实现内部必须先用 5 点关键点做仿射对齐再送入模型 ——
        漏掉对齐不会报错，但识别率会崩塌。
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


# ======================== 语音层（w 实现）========================

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
