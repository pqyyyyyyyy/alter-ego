"""感知业务逻辑的回归测试。

这层逻辑写错不会报错，只会表现成「立绘乱闪」或「角色认错人」，
而那时候你会先去怀疑模型和阈值。所以滞回与投票必须有测试钉住。

用假后端跑，不碰摄像头、不碰模型，毫秒级完成：
    python -m pytest tests/test_perception_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.contracts import FaceBox, Identity                 # noqa: E402
from alter_ego.perception.pipeline import (                       # noqa: E402
    PerceptionConfig,
    PerceptionPipeline,
)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
KPS = np.array([[40, 50], [70, 50], [55, 65], [43, 80], [67, 80]], dtype=np.float32)


def box(n_px: int = 100) -> FaceBox:
    return FaceBox(x1=0, y1=0, x2=n_px, y2=n_px, score=0.9, kps=KPS)


class FakeBackend:
    """按脚本返回人脸。embedding 用 user_id 编码，方便断言投票结果。"""

    def __init__(self, script: list[int], face_px: int = 100) -> None:
        self.script = script          # 每帧的人脸数
        self.face_px = face_px
        self.i = 0
        self.embed_calls = 0

    def detect_faces(self, frame):
        n = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return [box(self.face_px) for _ in range(n)]

    def extract_embedding(self, frame, b):
        self.embed_calls += 1
        v = np.zeros(512, dtype=np.float32)
        v[0] = 1.0
        return v


class FakeStore:
    """固定返回同一个身份，除非 answers 给了逐次脚本。"""

    def __init__(self, answers: list[Identity] | None = None) -> None:
        self.answers = answers
        self.i = 0

    def identify(self, emb):
        if self.answers is None:
            return Identity("u1", "小明", 0.72)
        a = self.answers[min(self.i, len(self.answers) - 1)]
        self.i += 1
        return a

    def enroll(self, name, embeddings):        # pragma: no cover
        raise AssertionError("识别流程绝不应该触发 enroll —— 这是隐私约定")

    def delete(self, user_id):                 # pragma: no cover
        raise AssertionError("识别流程不该删档")

    def list_users(self):
        return [("u1", "小明")]


def kinds(events) -> list[str]:
    return [e.kind for e in events]


@pytest.mark.parametrize('baseline', [False, True])
def test_scheduler_budget_pauses_features_but_keeps_presence_and_leave(baseline):
    from alter_ego.scheduler.coordinator import Coordinator
    coordinator = Coordinator(baseline=baseline)
    coordinator.reserve('chat', 'owner')
    backend = FakeBackend([1])
    pipe = PerceptionPipeline(backend, FakeStore(), PerceptionConfig(enter_frames=1, vote_samples=2))
    seen = []
    for _ in range(10):
        seen += kinds(pipe.step(FRAME, allow_identity=coordinator.budget().identity_extract))
    assert backend.i == 10 and 'APPEARED' in seen
    assert backend.embed_calls == (2 if baseline else 0)
    if not baseline:
        assert pipe.snapshot.primary is None
        for _ in range(2): pipe.step(FRAME, allow_identity=True)
        assert backend.embed_calls == 2 and pipe.snapshot.primary.user_id == 'u1'
    backend.script = [0]
    seen = []
    for _ in range(15): seen += kinds(pipe.step(FRAME, allow_identity=False))
    assert 'LEFT' in seen and not pipe.snapshot.present


# ---------------- 滞回 ----------------

def test_入场需要连续命中才算有人():
    cfg = PerceptionConfig(enter_frames=3, exit_frames=5)
    pipe = PerceptionPipeline(FakeBackend([1]), None, cfg)

    assert kinds(pipe.step(FRAME)) == []          # 第 1 帧
    assert kinds(pipe.step(FRAME)) == []          # 第 2 帧
    assert pipe.snapshot.present is False
    assert kinds(pipe.step(FRAME)) == ["APPEARED"]
    assert pipe.snapshot.present is True


def test_单帧漏检不应把人判成离开():
    """检测器偶发漏检是常态。逐帧直判会让角色在 IDLE/NOTICED 间抽搐。"""
    cfg = PerceptionConfig(enter_frames=2, exit_frames=5)
    # 有有有 漏 有有 —— 中间那次漏检不该触发 LEFT
    pipe = PerceptionPipeline(FakeBackend([1, 1, 1, 0, 1, 1]), None, cfg)

    seen = []
    for _ in range(6):
        seen += kinds(pipe.step(FRAME))

    assert "LEFT" not in seen
    assert pipe.snapshot.present is True


def test_连续漏检够久才算离开():
    cfg = PerceptionConfig(enter_frames=2, exit_frames=4)
    pipe = PerceptionPipeline(FakeBackend([1, 1, 0, 0, 0, 0]), None, cfg)

    seen = []
    for _ in range(6):
        seen += kinds(pipe.step(FRAME))

    assert seen.count("LEFT") == 1
    assert pipe.snapshot.present is False
    assert pipe.snapshot.primary is None


def test_出场阈值必须明显大于入场阈值():
    """代价不对称：误判「来了」只是提前打招呼，误判「走了」会打断对话。"""
    cfg = PerceptionConfig()
    assert cfg.exit_frames >= cfg.enter_frames * 3


def test_摄像头掉线应能回到IDLE():
    """step(None) 当作漏检处理，否则摄像头掉线时角色会卡在对话状态。"""
    cfg = PerceptionConfig(enter_frames=1, exit_frames=3)
    pipe = PerceptionPipeline(FakeBackend([1]), None, cfg)
    pipe.step(FRAME)
    assert pipe.snapshot.present is True

    seen = []
    for _ in range(3):
        seen += kinds(pipe.step(None))
    assert "LEFT" in seen


# ---------------- 身份投票 ----------------

def test_投票凑够样本才出结果():
    cfg = PerceptionConfig(enter_frames=1, vote_samples=3, vote_max_frames=10)
    be = FakeBackend([1])
    pipe = PerceptionPipeline(be, FakeStore(), cfg)

    assert kinds(pipe.step(FRAME)) == ["APPEARED"]   # 这一帧也投了一票
    assert kinds(pipe.step(FRAME)) == []
    ev = pipe.step(FRAME)
    assert kinds(ev) == ["IDENTIFIED"]
    assert ev[0].identity is not None
    assert ev[0].identity.name == "小明"


def test_认出之后不再反复提特征():
    """提特征只在身份可能变化时做。白烧 CPU 在板子上直接体现为语音卡顿。"""
    cfg = PerceptionConfig(enter_frames=1, vote_samples=3)
    be = FakeBackend([1])
    pipe = PerceptionPipeline(be, FakeStore(), cfg)

    for _ in range(3):
        pipe.step(FRAME)
    calls_after_vote = be.embed_calls

    for _ in range(20):
        pipe.step(FRAME)

    assert be.embed_calls == calls_after_vote, "投票结束后不应再提特征"


def test_人数变化触发重新识别():
    cfg = PerceptionConfig(enter_frames=1, vote_samples=2)
    pipe = PerceptionPipeline(FakeBackend([1, 1, 1, 2]), FakeStore(), cfg)

    seen = []
    for _ in range(4):
        seen += kinds(pipe.step(FRAME))

    assert "COUNT_CHANGED" in seen


def test_脸太小不提特征():
    """远处的小脸对齐后全是插值糊出来的，特征不可信且不会报错。"""
    cfg = PerceptionConfig(enter_frames=1, vote_samples=2,
                           vote_max_frames=3, min_face_px=60)
    be = FakeBackend([1], face_px=30)       # 30px < 60px 门限
    pipe = PerceptionPipeline(be, FakeStore(), cfg)

    seen = []
    for _ in range(4):
        seen += kinds(pipe.step(FRAME))

    assert be.embed_calls == 0
    assert "IDENTIFIED" in seen             # 超时结算，判访客
    assert pipe.snapshot.primary is not None
    assert pipe.snapshot.primary.user_id is None


def test_多数票定身份():
    cfg = PerceptionConfig(enter_frames=1, vote_samples=5)
    answers = [
        Identity("u1", "小明", 0.70),
        Identity("u1", "小明", 0.81),
        Identity(None, None, 0.30),           # 一帧误判
        Identity("u1", "小明", 0.65),
        Identity("u1", "小明", 0.60),
    ]
    pipe = PerceptionPipeline(FakeBackend([1]), FakeStore(answers), cfg)

    ident = None
    for _ in range(5):
        for ev in pipe.step(FRAME):
            if ev.kind == "IDENTIFIED":
                ident = ev.identity
    assert ident is not None
    assert ident.name == "小明"
    # 相似度取该身份票里最高的，不取平均 —— 平均会被姿态差的帧拖低
    assert ident.similarity == pytest.approx(0.81)


def test_平票判访客不蒙():
    cfg = PerceptionConfig(enter_frames=1, vote_samples=4)
    answers = [
        Identity("u1", "小明", 0.55),
        Identity("u2", "小红", 0.54),
        Identity("u1", "小明", 0.52),
        Identity("u2", "小红", 0.53),
    ]
    pipe = PerceptionPipeline(FakeBackend([1]), FakeStore(answers), cfg)

    ident = None
    for _ in range(4):
        for ev in pipe.step(FRAME):
            if ev.kind == "IDENTIFIED":
                ident = ev.identity
    assert ident is not None
    assert ident.user_id is None, "2:2 平票时叫名字就是赌，应判访客"


def test_纯检测降级模式不碰档案库():
    """store 为 None 时只判有人没人，一次特征都不提。"""
    cfg = PerceptionConfig(enter_frames=1)
    be = FakeBackend([1])
    pipe = PerceptionPipeline(be, None, cfg)

    for _ in range(10):
        pipe.step(FRAME)

    assert be.embed_calls == 0
    assert pipe.snapshot.present is True
    assert pipe.snapshot.primary is None


# ---------------- 视觉上下文 ----------------

def test_视觉上下文随状态变化():
    cfg = PerceptionConfig(enter_frames=1, vote_samples=1)
    pipe = PerceptionPipeline(FakeBackend([0]), None, cfg)
    pipe.step(FRAME)
    assert pipe.snapshot.visual_hint == "画面里没有人"

    pipe = PerceptionPipeline(FakeBackend([1]), FakeStore(), cfg)
    pipe.step(FRAME)
    assert "小明" in pipe.snapshot.visual_hint


def test_视觉上下文要短():
    """它跟人格卡 profile 一起进 system prompt，写长了挤占预算。"""
    cfg = PerceptionConfig(enter_frames=1, vote_samples=1)
    for script, store in (([0], None), ([1], FakeStore()), ([3], FakeStore())):
        pipe = PerceptionPipeline(FakeBackend(script), store, cfg)
        pipe.step(FRAME)
        assert len(pipe.snapshot.visual_hint) <= 30, pipe.snapshot.visual_hint
