"""档案库的回归测试，重点是隐私约定。

隐私这几条是赛事科技伦理审查项，口头声明不算，要有测试证明：
  · 只存特征向量，磁盘上不出现图像文件
  · identify 是纯只读的，不会自动建档
  · delete 把磁盘文件物理删除

    python -m pytest tests/test_identity_store.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.perception.identity import FileIdentityStore        # noqa: E402

rng = np.random.default_rng(42)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def person(seed: int, n: int = 5, jitter: float = 0.15) -> list[np.ndarray]:
    """造一个人的 n 个样本：同一基向量加小扰动，模拟不同角度。"""
    g = np.random.default_rng(seed)
    base = unit(g.normal(size=512))
    return [unit(base + jitter * g.normal(size=512)) for _ in range(n)]


@pytest.fixture()
def store(tmp_path: Path) -> FileIdentityStore:
    return FileIdentityStore(tmp_path / "users", threshold=0.4, margin=0.05)


# ---------------- 基本功能 ----------------

def test_注册后能认出本人(store):
    samples = person(1)
    store.enroll("小明", samples)

    ident = store.identify(samples[0])
    assert ident.user_id is not None
    assert ident.name == "小明"
    assert ident.similarity > 0.4


def test_陌生人判访客而不是最相近的那个(store):
    store.enroll("小明", person(1))

    ident = store.identify(unit(rng.normal(size=512)))
    assert ident.user_id is None
    assert ident.name is None


def test_空库不报错只返回访客(store):
    ident = store.identify(unit(rng.normal(size=512)))
    assert ident.user_id is None
    assert ident.similarity == 0.0


def test_样本不足应拒绝注册(store):
    with pytest.raises(ValueError, match="至少需要"):
        store.enroll("小明", person(1, n=2))


def test_两个人互不混淆(store):
    a, b = person(1), person(2)
    store.enroll("小明", a)
    store.enroll("小红", b)

    assert store.identify(a[0]).name == "小明"
    assert store.identify(b[0]).name == "小红"


def test_相似度过近时判认不准(store):
    """两人都在阈值上方且差距小于 margin 时，叫名字就是赌。"""
    base = unit(rng.normal(size=512))
    # 造两个几乎一样的档案 —— 真实场景里是双胞胎或采集条件退化
    g = np.random.default_rng(7)
    a = [unit(base + 0.02 * g.normal(size=512)) for _ in range(5)]
    b = [unit(base + 0.02 * g.normal(size=512)) for _ in range(5)]
    store.enroll("甲", a)
    store.enroll("乙", b)

    ident = store.identify(base)
    assert ident.user_id is None, "差距小于 margin 时应判访客"
    assert ident.similarity > 0.4, "但相似度本身是高的，要如实返回"


# ---------------- 持久化 ----------------

def test_重新加载后档案仍在(tmp_path):
    root = tmp_path / "users"
    samples = person(3)

    s1 = FileIdentityStore(root)
    uid = s1.enroll("小明", samples)

    s2 = FileIdentityStore(root)
    assert s2.list_users() == [(uid, "小明")]
    assert s2.identify(samples[0]).name == "小明"


def test_模板是归一化均值(store):
    samples = person(4, n=5)
    uid = store.enroll("小明", samples)
    rec = store.get(uid)

    assert rec is not None
    assert rec.template.shape == (512,)
    assert np.linalg.norm(rec.template) == pytest.approx(1.0, abs=1e-5)
    assert rec.samples.shape == (5, 512)


def test_注册时混入别人的脸应告警(store, caplog):
    """模板会变成两人的中点 —— 谁都认不出，且不会报错。"""
    mixed = person(1, n=3) + person(2, n=3)
    with caplog.at_level("WARNING"):
        store.enroll("混的", mixed)
    assert any("离散度偏高" in r.message for r in caplog.records)


# ---------------- 隐私约定 ----------------

def test_磁盘上只有特征和索引没有图像(store, tmp_path):
    store.enroll("小明", person(1))

    files = sorted(p.name for p in (tmp_path / "users").iterdir())
    assert any(f.endswith(".npz") for f in files)
    assert "users.json" in files
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    assert not [f for f in files if Path(f).suffix.lower() in image_ext]


def test_identify不会自动建档(store):
    """未注册者一律按匿名访客处理，不建档、不留痕。"""
    store.enroll("小明", person(1))
    before = store.list_users()

    for _ in range(20):
        store.identify(unit(rng.normal(size=512)))

    assert store.list_users() == before


def test_delete把磁盘文件物理删除(store, tmp_path):
    uid = store.enroll("小明", person(1))
    vec = tmp_path / "users" / f"{uid}.npz"
    assert vec.exists()

    store.delete(uid)

    assert not vec.exists(), "特征文件必须物理删除，不能只摘索引"
    assert store.list_users() == []
    index = json.loads((tmp_path / "users" / "users.json").read_text(encoding="utf-8"))
    assert index["users"] == {}


def test_delete是幂等的(store):
    store.delete("不存在的id")        # 不应抛异常
    uid = store.enroll("小明", person(1))
    store.delete(uid)
    store.delete(uid)


def test_删除后认不出该用户(store):
    samples = person(1)
    uid = store.enroll("小明", samples)
    store.delete(uid)

    ident = store.identify(samples[0])
    assert ident.user_id is None


# ---------------- 容错 ----------------

def test_特征文件缺失时告警而不是崩(tmp_path, caplog):
    root = tmp_path / "users"
    s1 = FileIdentityStore(root)
    uid = s1.enroll("小明", person(1))
    (root / f"{uid}.npz").unlink()

    with caplog.at_level("WARNING"):
        s2 = FileIdentityStore(root)
    assert any("特征文件缺失" in r.message for r in caplog.records)
    assert s2.list_users() == []


def test_索引损坏时按空库处理(tmp_path, caplog):
    root = tmp_path / "users"
    root.mkdir(parents=True)
    (root / "users.json").write_text("{ 这不是 json", encoding="utf-8")

    with caplog.at_level("ERROR"):
        s = FileIdentityStore(root)
    assert s.list_users() == []
