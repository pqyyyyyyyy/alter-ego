"""人脸注册：从摄像头采多角度样本，建立用户档案。

**注册必须由用户显式运行本脚本。** 识别流程（pipeline.py）是纯只读的，
没有任何代码路径会自动建档 —— 这是隐私设计的一部分，见 identity.py 顶部说明。

用法：
    python scripts/enroll_face.py --name 小明
    python scripts/enroll_face.py --list
    python scripts/enroll_face.py --delete <user_id>

采样时请按提示换角度（正脸、略左、略右、略低头、略抬头）。
单角度样本做出的模板对姿态过敏感，换个角度相似度就掉下阈值。

脚本会把对齐后的 112×112 小图存到 --dump-dir（默认 data/aligned_check/）。
**一定去看一眼这些图。** 正确的结果是人脸正立、居中、两眼在同一高度。
脸歪或偏就是对齐错了 —— 这个错误不报异常，只表现为识别率莫名很低。
注册完确认无误后可以把这个目录删掉（里面是人脸图，不入库、已 gitignore）。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.perception import Camera, FileIdentityStore          # noqa: E402
from alter_ego.perception.backends.insightface_backend import (     # noqa: E402
    InsightFaceBackend,
)

_log = logging.getLogger("enroll")

POSES = [
    "正对镜头",
    "头略微向左转",
    "头略微向右转",
    "略微低头",
    "略微抬头",
    "正对镜头，换一个表情（微笑）",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="人脸注册")
    ap.add_argument("--name", help="要注册的姓名")
    ap.add_argument("--samples", type=int, default=6, help="采样张数（默认 6）")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "users"))
    ap.add_argument("--dump-dir", default=str(ROOT / "data" / "aligned_check"))
    ap.add_argument("--min-face-px", type=int, default=100,
                    help="低于此像素宽的脸不收，太小的脸特征不可信")
    ap.add_argument("--list", action="store_true", help="列出已注册用户")
    ap.add_argument("--delete", metavar="USER_ID", help="删除指定用户")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store = FileIdentityStore(args.data_dir)

    if args.list:
        users = store.list_users()
        if not users:
            print("档案库为空")
        for uid, name in users:
            rec = store.get(uid)
            n = rec.n_samples if rec else "?"
            print(f"  {uid}  {name}  ({n} 张样本)")
        return 0

    if args.delete:
        store.delete(args.delete)
        return 0

    if not args.name:
        ap.error("要注册请给 --name，或用 --list / --delete")

    existing = store.find_by_name(args.name)
    if existing:
        print(f"已存在同名用户 {args.name} ({existing.user_id})。"
              f"先 --delete {existing.user_id} 再重新注册。")
        return 1

    dump = Path(args.dump_dir)
    dump.mkdir(parents=True, exist_ok=True)

    backend = InsightFaceBackend()
    embeddings = []

    print(f"\n准备为「{args.name}」采集 {args.samples} 张样本。")
    print("每次提示后保持姿势，脚本会自动抓取。按 Ctrl+C 中止。\n")

    with Camera(args.camera) as cam:
        for i in range(args.samples):
            pose = POSES[i % len(POSES)]
            print(f"[{i + 1}/{args.samples}] 请{pose} …", end="", flush=True)
            time.sleep(1.2)

            emb, note = _grab_one(backend, cam, args.min_face_px, dump, i)
            if emb is None:
                print(f" 失败（{note}），重试这一张")
                # 不计入计数，重试同一张
                if _retry(backend, cam, args.min_face_px, dump, i, embeddings):
                    print("      重试成功")
                else:
                    print("      重试仍失败，跳过")
                continue

            embeddings.append(emb)
            print(f" 收到（{note}）")

    if len(embeddings) < 3:
        print(f"\n只收到 {len(embeddings)} 张有效样本，不足 3 张，放弃注册。"
              "检查光线和人脸距离（脸宽至少 100 像素）。")
        return 1

    uid = store.enroll(args.name, embeddings)
    print(f"\n注册完成：{args.name} → {uid}，{len(embeddings)} 张样本")
    print(f"请打开 {dump} 肉眼确认对齐图：人脸应正立居中、两眼同高。")
    print("确认后可删除该目录（里面是人脸图，不入库）。")
    print("\n下一步：python scripts/bench_perception.py similarity   # 实测阈值")
    return 0


def _grab_one(backend, cam, min_px: int, dump: Path, idx: int):
    """抓一帧并提特征。返回 (embedding | None, 说明)。"""
    frame = cam.read_blocking(timeout=2.0)
    if frame is None:
        return None, "没拿到画面"

    faces = backend.detect_faces(frame)
    if not faces:
        return None, "没检测到人脸"
    if len(faces) > 1:
        return None, f"画面里有 {len(faces)} 张脸，注册时请只保留一人"

    face = faces[0]
    w = face.x2 - face.x1
    if w < min_px:
        return None, f"脸太小（{w}px < {min_px}px），请靠近镜头"
    if face.kps is None:
        return None, "检测器没给关键点，无法对齐"

    backend.dump_aligned(frame, face, str(dump / f"{idx:02d}.png"))
    return backend.extract_embedding(frame, face), f"脸宽 {w}px，置信度 {face.score:.2f}"


def _retry(backend, cam, min_px: int, dump: Path, idx: int, out: list) -> bool:
    for _ in range(3):
        time.sleep(1.0)
        emb, _note = _grab_one(backend, cam, min_px, dump, idx)
        if emb is not None:
            out.append(emb)
            return True
    return False


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中止，未写入任何档案")
        raise SystemExit(130)
