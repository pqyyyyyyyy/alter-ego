"""摄像头采集：独立线程抓帧，只保留最新一帧。

**为什么必须丢帧而不是排队**

`cv2.VideoCapture.read()` 背后有缓冲队列。如果推理比相机帧率慢（板子上
一定慢），每次 read 拿到的都是队列头部的旧帧，延迟会一帧一帧累积 ——
表现是角色对着你几秒钟前的动作反应，看起来像「卡」，但 CPU 占用不高，
很难归因。

解法是采集线程全速 read 并覆盖写入单槽，消费端永远拿到最新帧。
旧帧直接丢掉 —— 感知不需要完整视频流，只需要「现在画面里是什么」。
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

_log = logging.getLogger(__name__)


class Camera:
    """最新帧优先的摄像头。用作上下文管理器。

        with Camera(0) as cam:
            frame = cam.read()
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        warmup_sec: float = 1.0,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.warmup_sec = warmup_sec

        self._cap = None
        self._frame: np.ndarray | None = None
        self._seq = 0                       # 帧序号，消费端用它判断是否拿到新帧
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def open(self) -> None:
        import cv2

        # Windows 上默认后端打开摄像头要 2-3 秒，DSHOW 快很多。
        # 板子是 Linux，用默认后端（V4L2）。
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)       # 退回默认后端再试一次
        if not cap.isOpened():
            raise RuntimeError(
                f"打不开摄像头 {self.index}。检查是否被其他程序占用，"
                "或换一个 index（实测本机 0 和 1 都可用）。"
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # 缓冲设为 1 是双保险。多数后端不理这个属性，所以不能只靠它 ——
        # 采集线程的覆盖写才是真正的解法。
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        actual = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _log.info("摄像头 %d 已打开，请求 %dx%d，实际 %.0fx%.0f",
                  self.index, self.width, self.height, *actual)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()

        # 等第一帧。摄像头自动曝光需要时间，立刻抓的头几帧往往全黑或过暗，
        # 用它们做检测会漏检 —— 这个坑不报错，只表现成「刚启动认不出人」。
        deadline = time.time() + max(self.warmup_sec, 0.3) + 3.0
        while time.time() < deadline:
            if self._frame is not None and self._seq > 3:
                break
            time.sleep(0.05)
        time.sleep(self.warmup_sec)

    def _loop(self) -> None:
        assert self._cap is not None
        fail = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fail += 1
                if fail > 50:
                    _log.error("连续 %d 次抓帧失败，采集线程退出", fail)
                    break
                time.sleep(0.02)
                continue
            fail = 0
            with self._lock:
                if self._frame is not None:
                    self.dropped += 1       # 上一帧还没被消费就被覆盖了
                self._frame = frame
                self._seq += 1

    def read(self) -> np.ndarray | None:
        """取最新一帧（BGR）。取走后清空槽位，便于统计丢帧。"""
        with self._lock:
            frame = self._frame
            self._frame = None
        return frame

    def read_blocking(self, timeout: float = 2.0) -> np.ndarray | None:
        """等到有新帧为止。基准测试用，正式链路用非阻塞的 read()。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    @property
    def seq(self) -> int:
        return self._seq

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        _log.info("摄像头已关闭，共抓 %d 帧，丢弃 %d 帧", self._seq, self.dropped)

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
