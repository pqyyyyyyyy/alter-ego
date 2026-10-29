"""用户档案库：人脸特征的注册、比对与删除。

**隐私设计（赛事规则的科技伦理审查项，主动做是加分项）**

  · 只存 512 维特征向量，**不存任何原始人脸图像**。特征向量仍属敏感生物特征数据，不能承诺不可逆。
  · 注册必须由用户显式触发（`enroll`），没有任何代码路径会在识别过程中
    自动建档 —— `identify` 是纯只读的。
  · `delete()` 删干净：磁盘上的 .npz 文件物理删除，索引里的条目一并移除。
  · 未注册者一律按匿名访客处理（`Identity.user_id is None`），不建档、不留痕。
  · 全部数据落在 `data/users/`（已 gitignore），不出设备、不上传。

**阈值不能拍脑袋**

余弦相似度的判定阈值依赖模型与采集条件，必须实测：测本人多次出现的
相似度（应 0.6+）和他人的相似度（应 0.2-），阈值取两组数字中间。
用 `scripts/bench_perception.py` 测，数字记进 docs/实验数据.md。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import EMBEDDING_DIM, Identity, IdentityStore

_log = logging.getLogger(__name__)

# 默认阈值。**这是占位值，必须用 bench_perception.py 实测后改掉。**
# 留 0.40 是 ArcFace 系列的常见中位数，但 w600k_mbf 比 w600k_r50 小，
# 类间间隔更窄，实测值很可能不同。
DEFAULT_THRESHOLD = 0.40

# 第一名与第二名的最小差距。只过阈值不够 ——
# 两个人都 0.45 时选谁都是赌，这种情况应判为「认不准」而不是蒙一个。
# 家用场景档案数只有 2-5 个，这条规则代价极低，收益是不会叫错名字。
DEFAULT_MARGIN = 0.05

MIN_ENROLL_SAMPLES = 3


@dataclass
class UserRecord:
    """一个用户的档案。template 是 samples 的归一化均值。"""
    user_id: str
    name: str
    template: np.ndarray          # (512,) float32，L2 归一化
    samples: np.ndarray           # (N, 512) float32，保留用于重算阈值
    created_at: float

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])


class FileIdentityStore(IdentityStore):
    """落盘实现：索引走 JSON，特征走 .npz。

    为什么不用数据库：档案量级是个位数，JSON + npz 零依赖、可手工检查、
    删除就是删文件 —— 对「删干净」这个隐私要求来说，文件系统比数据库更好验证。
    """

    INDEX_NAME = "users.json"

    def __init__(
        self,
        root: str | Path = "data/users",
        threshold: float = DEFAULT_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.margin = margin
        self._users: dict[str, UserRecord] = {}
        self._load()

    # ---------------- 持久化 ----------------

    @property
    def _index_path(self) -> Path:
        return self.root / self.INDEX_NAME

    def _vec_path(self, user_id: str) -> Path:
        return self.root / f"{user_id}.npz"

    def _load(self) -> None:
        if not self._index_path.exists():
            _log.info("档案库为空: %s", self.root)
            return
        try:
            index = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log.error("索引读取失败，按空库处理: %s", exc)
            return

        for uid, meta in index.get("users", {}).items():
            path = self._vec_path(uid)
            if not path.exists():
                # 索引有、特征文件没了。不自动清理索引 ——
                # 让它留着报警，比悄悄抹掉更容易发现是哪一步删漏了。
                _log.warning("用户 %s 的特征文件缺失: %s", uid, path)
                continue
            with np.load(path) as z:
                samples = z["samples"].astype(np.float32)
                template = z["template"].astype(np.float32)
            self._users[uid] = UserRecord(
                user_id=uid, name=meta["name"], template=template,
                samples=samples, created_at=meta.get("created_at", 0.0),
            )
        _log.info("已载入 %d 个用户档案: %s",
                  len(self._users), [u.name for u in self._users.values()])

    def _save_index(self) -> None:
        payload = {
            "version": 1,
            "embedding_dim": EMBEDDING_DIM,
            "threshold": self.threshold,
            "users": {
                u.user_id: {
                    "name": u.name,
                    "created_at": u.created_at,
                    "n_samples": u.n_samples,
                }
                for u in self._users.values()
            },
        }
        temp = self._index_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(self._index_path)

    # ---------------- IdentityStore ----------------

    def enroll(self, name: str, embeddings: list[np.ndarray]) -> str:
        """注册新用户。需用户显式触发，不会被识别流程自动调用。

        多张不同角度的样本取归一化均值作模板 —— 单张样本的模板对姿态
        过敏感，换个角度相似度就掉下阈值。
        """
        if len(embeddings) < MIN_ENROLL_SAMPLES:
            raise ValueError(
                f"至少需要 {MIN_ENROLL_SAMPLES} 张样本，收到 {len(embeddings)} 张。"
                "样本太少时模板对姿态过敏感，识别会不稳。"
            )

        samples = np.stack([self._normalize(e) for e in embeddings]).astype(np.float32)
        # 归一化向量的均值是质心，再归一化一次让它回到单位球面上，
        # 这样余弦相似度仍然等价于点积。
        template = self._normalize(samples.mean(axis=0))

        # 样本自洽性检查。注册时若把两个人的脸混进同一批，
        # 模板会变成两人的中点 —— 谁都认不出，且不会报错。
        spread = float(np.min(samples @ template))
        if spread < 0.5:
            _log.warning(
                "注册样本离散度偏高（最低样本-模板相似度 %.3f）。"
                "可能混入了别人的脸，或采集时光线/角度差异过大。", spread,
            )

        user_id = uuid.uuid4().hex[:12]
        rec = UserRecord(
            user_id=user_id, name=name, template=template,
            samples=samples, created_at=time.time(),
        )
        np.savez(self._vec_path(user_id), template=template, samples=samples)
        self._users[user_id] = rec
        try:
            self._save_index()
        except Exception:
            self._users.pop(user_id, None)
            self._vec_path(user_id).unlink(missing_ok=True)
            raise
        _log.info("已注册 %s (%s)，%d 张样本，自洽度 %.3f",
                  name, user_id, rec.n_samples, spread)
        return user_id

    def identify(self, embedding: np.ndarray) -> Identity:
        """只读比对。认不出就返回匿名访客，绝不建档。"""
        if not self._users:
            return Identity(user_id=None, name=None, similarity=0.0)

        q = self._normalize(embedding)
        scored = sorted(
            ((float(q @ u.template), u) for u in self._users.values()),
            key=lambda t: t[0], reverse=True,
        )
        best_sim, best = scored[0]
        second_sim = scored[1][0] if len(scored) > 1 else -1.0

        if best_sim < self.threshold:
            return Identity(user_id=None, name=None, similarity=best_sim)
        if best_sim - second_sim < self.margin:
            _log.info("两人相似度过近（%.3f vs %.3f），判为认不准",
                      best_sim, second_sim)
            return Identity(user_id=None, name=None, similarity=best_sim)
        return Identity(user_id=best.user_id, name=best.name, similarity=best_sim)

    def delete(self, user_id: str) -> None:
        """删除用户全部数据，包括磁盘文件。幂等。"""
        rec = self._users.get(user_id)
        path = self._vec_path(user_id)
        tomb = path.with_suffix('.npz.deleting')
        if path.exists(): path.replace(tomb)
        self._users.pop(user_id, None)
        try:
            self._save_index()
        except Exception:
            if rec: self._users[user_id] = rec
            if tomb.exists(): tomb.replace(path)
            raise
        if tomb.exists(): tomb.unlink()
        if rec:
            _log.info("已删除用户 %s (%s)，特征文件已物理删除", rec.name, user_id)
        else:
            _log.info("用户 %s 不存在，无需删除", user_id)

    def list_users(self) -> list[tuple[str, str]]:
        return [(u.user_id, u.name) for u in self._users.values()]

    # ---------------- 辅助 ----------------

    def get(self, user_id: str) -> UserRecord | None:
        return self._users.get(user_id)

    def find_by_name(self, name: str) -> UserRecord | None:
        for u in self._users.values():
            if u.name == name:
                return u
        return None

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        if v.shape != (EMBEDDING_DIM,) or not np.isfinite(v).all():
            raise ValueError('embedding must be finite shape (512,)')
        n = float(np.linalg.norm(v))
        if n <= 1e-8: raise ValueError('zero embedding norm')
        return v / n
