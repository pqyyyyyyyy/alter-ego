"""感知链路的业务层：检测 → 在场判定 → 身份识别 → 事件。

本模块不碰任何模型，只做判断。模型全在 `backends/` 后端里，
所以这层逻辑在 PC 和板子上是同一份代码，上板只换后端。

**两个必须做对的设计**

一、**在场判定要有滞回（hysteresis），不能逐帧直判。**
   检测器偶发漏检是常态（侧脸、眨眼、运动模糊）。逐帧直判的后果是
   角色在 IDLE 和 NOTICED 之间反复抽搐，立绘和状态疯狂闪烁。
   解法：连续 N 帧有人才算来了，连续 M 帧没人才算走了。
   出场阈值远大于入场阈值 —— 误判「来了」只是提前打招呼，
   误判「走了」会把正在说话的对话打断，代价不对称。

二、**提特征不能每帧都做。**
   检测 ~10 ms，提特征 ~15 ms，但提特征的收益只在身份变化时才有。
   板子 4 核要分给 ASR 和 TTS，白烧 CPU 直接体现为语音卡顿。
   解法：只在「有人新出现」或「人数变了」时才投票识别一次，
   认出来就缓存住，之后一直复用，直到人离开。

事件流给上层（server/app.py）用来驱动状态机：
   APPEARED  → IDLE 转 NOTICED，按身份决定打招呼还是招呼访客
   LEFT      → 回 IDLE
   IDENTIFIED→ 身份投票出结果，可更新 PersonaContext.user_name
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..contracts import FaceBox, Identity, IdentityStore, PerceptionBackend

_log = logging.getLogger(__name__)

EventKind = Literal["APPEARED", "LEFT", "IDENTIFIED", "COUNT_CHANGED"]


@dataclass
class PerceptionConfig:
    # 滞回帧数。入场 3 帧、出场 15 帧 —— 按 5 fps 算约为「0.6 秒认人、
    # 3 秒才算走」。3 秒容忍了低头、转身、起身拿东西这类常见动作。
    enter_frames: int = 3
    exit_frames: int = 15

    # 身份投票用的样本数。单帧识别受姿态影响大，多帧投票稳得多。
    vote_samples: int = 5
    # 投票时最多等多少帧还没凑够样本就按现有的结算，避免一直等不出结果
    vote_max_frames: int = 20

    # 空闲时降低检测频率。没人的时候没必要 5 fps 盯着，省下来的 CPU
    # 在板子上是实打实的 —— 这也是 C1 调度器可以调的旋钮。
    idle_fps: float = 2.0
    active_fps: float = 5.0

    # 太小的脸不提特征。远处路过的人脸只有几十像素，对齐后全是插值
    # 糊出来的，提出来的特征不可信 —— 它不会报错，只会污染识别结果。
    min_face_px: int = 60


@dataclass
class PerceptionEvent:
    kind: EventKind
    n_faces: int = 0
    identity: Identity | None = None
    detail: str = ""


@dataclass
class PerceptionSnapshot:
    """当前感知状态。server 可以随时查，也可以直接塞进 PersonaContext。"""
    present: bool = False
    n_faces: int = 0
    primary: Identity | None = None
    visual_hint: str = ""
    frames_seen: int = 0
    # 冷启动首帧含 ORT 图优化与内存池分配，实测 ~130 ms，是稳态的 50 倍。
    # 只报均值会被它拉高（26 帧时均值 7.6 ms 而稳态 2.7 ms），
    # 排查时看着像「检测很慢」。所以两个都报，诊断看中位数。
    detect_mean_ms: float = 0.0
    detect_median_ms: float = 0.0


@dataclass
class _VoteState:
    """一轮身份投票的中间状态。"""
    votes: list[Identity] = field(default_factory=list)
    frames: int = 0

    def reset(self) -> None:
        self.votes.clear()
        self.frames = 0


class PerceptionPipeline:
    """把后端与档案库串成「画面 → 事件」。

    用法是每拿到一帧就 `step(frame)`，拿回这一帧引发的事件列表。
    不传帧也可以调 `step(None)` —— 会被当作一次漏检计入滞回，
    这样摄像头掉线时角色会正常回到 IDLE 而不是卡在对话状态。
    """

    def __init__(
        self,
        backend: PerceptionBackend,
        store: IdentityStore | None = None,
        config: PerceptionConfig | None = None,
    ) -> None:
        self.backend = backend
        self.store = store
        self.cfg = config or PerceptionConfig()

        self.snapshot = PerceptionSnapshot()
        self._hit_streak = 0            # 连续有人的帧数
        self._miss_streak = 0           # 连续没人的帧数
        self._vote = _VoteState()
        self._voting = False
        self._last_n_faces = 0
        self._detect_ms: list[float] = []

    # ---------------- 主循环 ----------------

    @property
    def target_fps(self) -> float:
        """当前该用的检测频率。调用方据此 sleep。"""
        return self.cfg.active_fps if self.snapshot.present else self.cfg.idle_fps

    def step(self, frame: np.ndarray | None, allow_identity: bool = True) -> list[PerceptionEvent]:
        events: list[PerceptionEvent] = []
        self.snapshot.frames_seen += 1

        faces: list[FaceBox] = []
        if frame is not None:
            t0 = time.perf_counter()
            faces = self.backend.detect_faces(frame)
            self._detect_ms.append((time.perf_counter() - t0) * 1000)
            if len(self._detect_ms) > 200:
                del self._detect_ms[:-200]
            self.snapshot.detect_mean_ms = round(
                sum(self._detect_ms) / len(self._detect_ms), 2
            )
            ordered = sorted(self._detect_ms)
            self.snapshot.detect_median_ms = round(
                ordered[len(ordered) // 2], 2
            )

        n = len(faces)

        # ---- 滞回：更新在场状态 ----
        if n > 0:
            self._hit_streak += 1
            self._miss_streak = 0
        else:
            self._miss_streak += 1
            self._hit_streak = 0

        if not self.snapshot.present and self._hit_streak >= self.cfg.enter_frames:
            self.snapshot.present = True
            self.snapshot.n_faces = n
            self._last_n_faces = n
            self._start_vote()
            events.append(PerceptionEvent("APPEARED", n_faces=n))
            _log.info("有人出现，%d 张脸", n)

        elif self.snapshot.present and self._miss_streak >= self.cfg.exit_frames:
            self.snapshot.present = False
            self.snapshot.n_faces = 0
            self.snapshot.primary = None
            self._last_n_faces = 0
            self._voting = False
            self._vote.reset()
            self.snapshot.visual_hint = self._render_hint()
            events.append(PerceptionEvent("LEFT"))
            _log.info("人已离开（连续 %d 帧无人）", self._miss_streak)
            return events

        if not self.snapshot.present:
            self.snapshot.visual_hint = self._render_hint()
            return events

        # ---- 在场期间：人数变化要重新识别 ----
        if n > 0 and n != self._last_n_faces:
            self._last_n_faces = n
            self.snapshot.n_faces = n
            self.snapshot.primary = None
            self._start_vote()
            events.append(PerceptionEvent("COUNT_CHANGED", n_faces=n))
            _log.info("画面人数变为 %d，重新识别", n)

        # Without tracking, several faces cannot identify the active speaker safely.
        if n > 1:
            self._voting = False
            self.snapshot.primary = Identity(None, None, 0.0)

        # ---- 身份投票（只在需要时跑，这是省 CPU 的关键）----
        if allow_identity and self._voting and n > 0:
            ev = self._vote_once(frame, faces[0])
            if ev is not None:
                events.append(ev)

        self.snapshot.visual_hint = self._render_hint()
        return events

    # ---------------- 身份识别 ----------------

    def _start_vote(self) -> None:
        if self.store is None:
            return          # 纯检测降级模式，不识别
        self._vote.reset()
        self._voting = True

    def _vote_once(
        self, frame: np.ndarray | None, face: FaceBox
    ) -> PerceptionEvent | None:
        """给主脸投一票。凑够样本或超时后结算。"""
        if frame is None or self.store is None:
            return None

        self._vote.frames += 1

        w, h = face.x2 - face.x1, face.y2 - face.y1
        if (face.kps is not None and np.asarray(face.kps).shape == (5, 2)
            and np.isfinite(face.kps).all() and min(w, h) >= self.cfg.min_face_px):
            try:
                emb = np.asarray(self.backend.extract_embedding(frame, face), dtype=np.float32)
                if emb.shape != (512,) or not np.isfinite(emb).all() or np.linalg.norm(emb) <= 1e-8:
                    raise ValueError('invalid embedding')
                emb = emb / np.linalg.norm(emb)
                self._vote.votes.append(self.store.identify(emb))
            except Exception as exc:
                # 提特征失败不能让整条链路断掉，但必须留下日志。
                # 容错和可观测性是两件事，不能用前者替代后者。
                _log.warning("提取特征失败，这一票作废: %s", exc)

        enough = len(self._vote.votes) >= self.cfg.vote_samples
        timeout = self._vote.frames >= self.cfg.vote_max_frames
        if not (enough or timeout):
            return None

        self._voting = False
        if not self._vote.votes:
            _log.info("投票期内一票未收到（脸太小或对齐失败），按访客处理")
            self.snapshot.primary = Identity(None, None, 0.0)
            return PerceptionEvent(
                "IDENTIFIED", n_faces=self.snapshot.n_faces,
                identity=self.snapshot.primary, detail="无有效样本",
            )

        ident = self._tally(self._vote.votes)
        self.snapshot.primary = ident
        _log.info("身份投票结果: %s (相似度 %.3f)，%d 票",
                  ident.name or "访客", ident.similarity, len(self._vote.votes))
        return PerceptionEvent(
            "IDENTIFIED", n_faces=self.snapshot.n_faces, identity=ident,
            detail=f"{len(self._vote.votes)} 票",
        )

    @staticmethod
    def _tally(votes: list[Identity]) -> Identity:
        """多数票定身份，相似度取该身份的票里最高的那个。

        不用平均相似度 —— 平均会被姿态差的那几帧拖低，而我们关心的是
        「最好的一次有多像」。多数票本身已经抑制了偶发误判。
        """
        counts = Counter(v.user_id for v in votes)
        winner_id, n_win = counts.most_common(1)[0]
        # 平票时（家用场景档案少，两人各 2 票是可能的）判访客，不蒙
        if list(counts.values()).count(n_win) > 1:
            best = max((v.similarity for v in votes), default=0.0)
            return Identity(None, None, best)
        same = [v for v in votes if v.user_id == winner_id]
        best = max(same, key=lambda v: v.similarity)
        return Identity(winner_id, best.name, best.similarity)

    # ---------------- 给人格层的视觉上下文 ----------------

    def _render_hint(self) -> str:
        """渲染成一句中文，塞进 PersonaContext.visual_hint 进云端 prompt。

        要短。它跟人格卡的 profile 一起进 system prompt，写长了挤占预算，
        而且云端模型会抓着细节编故事 —— 说「画面偏暗」它就会聊天黑了。
        """
        if not self.snapshot.present:
            return "画面里没有人"
        n = self.snapshot.n_faces
        p = self.snapshot.primary
        if n > 1:
            who = f"，其中{p.name}在最前面" if p and p.name else ""
            return f"画面里有 {n} 个人{who}"
        if p is None:
            return "画面里有一个人，还没认出是谁"
        if p.name:
            return f"画面里是{p.name}"
        return "画面里有一个人，不认识"

    # ---------------- 诊断 ----------------

    def stats(self) -> dict[str, object]:
        return {
            "present": self.snapshot.present,
            "n_faces": self.snapshot.n_faces,
            "primary": self.snapshot.primary.name if self.snapshot.primary else None,
            "similarity": round(self.snapshot.primary.similarity, 3)
                          if self.snapshot.primary else None,
            "frames_seen": self.snapshot.frames_seen,
            "detect_mean_ms": self.snapshot.detect_mean_ms,
            "detect_median_ms": self.snapshot.detect_median_ms,
            "visual_hint": self.snapshot.visual_hint,
            "voting": self._voting,
        }
