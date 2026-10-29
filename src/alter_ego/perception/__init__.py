"""感知链路：摄像头 → 人脸检测 → 身份识别 → 事件。

分层，上板时只换最底下一层：

    camera.py     采集（最新帧优先，丢旧帧）
    backends/     推理后端 —— PC 用 insightface，板子加 ascend_backend.py
    identity.py   档案库（只存特征，不存人脸图）
    pipeline.py   业务判断（滞回、身份投票、视觉上下文），与后端无关

后端不在这里导出 —— `import insightface` 要两秒多，而人格层和语音层
都不需要它。要用就显式写：

    from alter_ego.perception.backends.insightface_backend import InsightFaceBackend
"""
from __future__ import annotations

from .camera import Camera
from .identity import DEFAULT_MARGIN, DEFAULT_THRESHOLD, FileIdentityStore, UserRecord
from .pipeline import (
    PerceptionConfig,
    PerceptionEvent,
    PerceptionPipeline,
    PerceptionSnapshot,
)

__all__ = [
    "Camera",
    "FileIdentityStore",
    "UserRecord",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MARGIN",
    "PerceptionPipeline",
    "PerceptionConfig",
    "PerceptionEvent",
    "PerceptionSnapshot",
]
