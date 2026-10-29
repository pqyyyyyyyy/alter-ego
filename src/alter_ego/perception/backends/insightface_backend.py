"""PC 参照后端：把 insightface 的 SCRFD + ArcFace 包成 PerceptionBackend。

**为什么用现成实现而不手搓 ONNX 前后处理**

感知链路有两个静默失败点，都不报错、流程照跑、只是结果悄悄不对：

  1. SCRFD 的后处理是多尺度 anchor 解码 + NMS。自己推公式极易写错，
     而写错的表现是框偏移或漏检，不是异常。
  2. ArcFace 不能喂裁剪图，必须先用 5 点关键点做仿射对齐到 112×112。
     漏掉对齐同样不报错，但同一人两次的相似度会从 0.6 掉到 0.2 ——
     然后你会去怀疑阈值、模型、光线，卡一两天。

insightface 把这两处都封好了（许可证 MIT，已登记在 assets/manifest.yaml）。
赛事规则允许使用第三方开源代码，如实登记来源与许可即可。

**为什么不直接用 FaceAnalysis.get()**

`get()` 一次调用做完检测 + 对齐 + 提特征，但它把两步焊死了。我们需要分开：
检测每帧都要跑（判断有人没人），提特征只在需要认人时跑 —— 这是板子上
省 CPU 的关键，也是 C1 调度器能排布 NPU 任务的前提。
所以这里只借 FaceAnalysis 的模型下载与解压，拿到子模型后自己串。

**上板计划**（见 docs/方案/昇腾模型转换调研.md）

本文件不动，另写 `ascend_backend.py` 实现同一个 Protocol。届时检测换
YOLOv8-face（Ultralytics 原生支持 `export(format="ascend")`），
因为 SCRFD/ArcFace 在昇腾零公开成功记录，是全项目最大的移植风险点。
两个后端输出同一套 FaceBox / embedding，业务逻辑一行不改。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ...contracts import EMBEDDING_DIM, FaceBox, PerceptionBackend

_log = logging.getLogger(__name__)

# buffalo_s = det_500m（检测）+ w600k_mbf（识别），约 80 MB。
# 不用默认的 buffalo_l（326 MB，det_10g + w600k_r50）—— 板子只有 4 核，
# 大模型的精度收益换不回延迟，且 310B 显存与带宽都紧。
# PC 上两者都能跑，选小的是为了 PC 的数字对板子有参考价值。
DEFAULT_MODEL = "buffalo_s"

# 检测输入尺寸。640 是 insightface 默认，但角色设备是桌面近距离使用，
# 人脸在画面里占比大，320 足够且快一倍。实测数据见 docs/实验数据.md。
DEFAULT_DET_SIZE = (320, 320)


@dataclass
class BackendStats:
    """耗时统计。给 C1 调度器做基线，也直接进技术报告。"""
    detect_calls: int = 0
    detect_ms: list[float] = field(default_factory=list)
    embed_calls: int = 0
    embed_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float | int]:
        def stat(xs: list[float]) -> tuple[float, float]:
            if not xs:
                return 0.0, 0.0
            s = sorted(xs)
            return sum(xs) / len(xs), s[len(s) // 2]

        d_mean, d_med = stat(self.detect_ms)
        e_mean, e_med = stat(self.embed_ms)
        return {
            "detect_calls": self.detect_calls,
            "detect_mean_ms": round(d_mean, 2),
            "detect_median_ms": round(d_med, 2),
            "embed_calls": self.embed_calls,
            "embed_mean_ms": round(e_mean, 2),
            "embed_median_ms": round(e_med, 2),
        }


class InsightFaceBackend(PerceptionBackend):
    """SCRFD 检测 + ArcFace 特征，CPU ONNXRuntime。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        det_thresh: float = 0.5,
        num_threads: int = 0,
    ) -> None:
        """num_threads > 0 时限制 ORT 线程数。

        板子 4 核要分给 ASR 与 TTS，感知不能把核吃光 —— 实测线程数对
        速度的影响已量化（见 docs/实验数据.md 的 llama-bench 一节，
        2→4 线程差 50%），这个参数是调度器的旋钮之一。
        """
        from insightface.app import FaceAnalysis

        self.model_name = model_name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.stats = BackendStats()

        t0 = time.perf_counter()
        # allowed_modules 只留两个 —— 默认还会加载 genderage 与 2d106 关键点，
        # 我们用不到，加载它们白费内存和启动时间。
        app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_thresh=det_thresh, det_size=det_size)

        self._det = app.models["detection"]
        self._rec = app.models["recognition"]
        if num_threads > 0:
            self._limit_threads(num_threads)

        self.load_sec = time.perf_counter() - t0
        _log.info(
            "感知后端就绪: %s, det_size=%s, det_thresh=%.2f, 加载耗时 %.2fs",
            model_name, det_size, det_thresh, self.load_sec,
        )

    def _limit_threads(self, n: int) -> None:
        """重建 session 以套用线程数限制。

        ORT 的线程数只能在建 session 时指定，改不了已有的 session，
        所以这里拿原 session 的模型路径重建一遍。
        """
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n
        opts.inter_op_num_threads = 1
        for name, model in (("detection", self._det), ("recognition", self._rec)):
            path = getattr(model, "model_file", None)
            if not path:
                _log.warning("%s 拿不到模型路径，线程限制未生效", name)
                continue
            model.session = ort.InferenceSession(
                path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        # detection 重建 session 后要重新 prepare，否则 input_size 等
        # 推理期参数会丢（这个坑不报错，表现是 detect 直接抛 shape 错）
        self._det.prepare(-1, det_thresh=self.det_thresh, input_size=self.det_size)
        _log.info("ORT 线程数限制为 %d", n)

    # ---------------- PerceptionBackend ----------------

    def detect_faces(self, frame: np.ndarray) -> list[FaceBox]:
        """输入 BGR 图像，返回人脸框列表，按框面积从大到小排序。

        排序是业务约定：桌面场景下最大的脸就是正在跟角色说话的人，
        下游取 `faces[0]` 即可，不必再判断。
        """
        t0 = time.perf_counter()
        bboxes, kpss = self._det.detect(
            frame, input_size=self.det_size, max_num=0, metric="default"
        )
        self.stats.detect_ms.append((time.perf_counter() - t0) * 1000)
        self.stats.detect_calls += 1

        if bboxes is None or len(bboxes) == 0:
            return []

        out: list[FaceBox] = []
        for i, b in enumerate(bboxes):
            kps = None
            if kpss is not None and i < len(kpss):
                kps = np.asarray(kpss[i], dtype=np.float32)
            out.append(FaceBox(
                x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]),
                score=float(b[4]), kps=kps,
            ))
        out.sort(key=lambda f: (f.x2 - f.x1) * (f.y2 - f.y1), reverse=True)
        return out

    def extract_embedding(self, frame: np.ndarray, box: FaceBox) -> np.ndarray:
        """5 点仿射对齐 → 112×112 → ArcFace → L2 归一化。

        归一化在这里做掉，下游的余弦相似度就退化成点积。
        """
        from insightface.utils import face_align

        if box.kps is None:
            # 不静默兜底。没有关键点就对不了齐，返回的向量是垃圾，
            # 而垃圾向量不会报错 —— 它会一路流到相似度比对里，
            # 表现成「识别准确率莫名很差」。宁可在这里炸掉。
            raise ValueError(
                "FaceBox.kps 为 None，无法做仿射对齐。"
                "纯检测降级模式下不应调用 extract_embedding。"
            )

        t0 = time.perf_counter()
        aligned = face_align.norm_crop(frame, landmark=box.kps, image_size=112)
        emb = self._rec.get_feat(aligned).flatten().astype(np.float32)
        self.stats.embed_ms.append((time.perf_counter() - t0) * 1000)
        self.stats.embed_calls += 1

        if emb.shape[0] != EMBEDDING_DIM:
            _log.warning(
                "特征维度 %d ≠ 约定的 %d，IdentityStore 的历史档案会不兼容",
                emb.shape[0], EMBEDDING_DIM,
            )
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 1e-8 else emb

    # ---------------- 调试辅助 ----------------

    def dump_aligned(self, frame: np.ndarray, box: FaceBox, path: str) -> None:
        """把对齐结果存成图片，用来肉眼验证对齐是否正确。

        正确的结果是人脸正着、居中、两眼在固定高度。脸歪或偏就是对齐错了。
        这是检查静默失败点 ② 的唯一可靠办法 —— 相似度低时先看这张图，
        不要先去调阈值。
        """
        import cv2

        from insightface.utils import face_align

        if box.kps is None:
            raise ValueError("没有关键点，无从对齐")
        aligned = face_align.norm_crop(frame, landmark=box.kps, image_size=112)
        cv2.imwrite(path, aligned)
        _log.info("对齐结果已存到 %s，请肉眼确认人脸正立居中", path)
