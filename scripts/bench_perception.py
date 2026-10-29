"""感知链路基准测试：产出技术报告要的四组数字。

| 子命令 | 产出 |
|---|---|
| `latency`    | 检测 / 特征提取单帧耗时（det_size × 线程数 扫描） |
| `similarity` | 本人相似度 vs 他人相似度，阈值扫描（FAR/FRR） |
| `pairs`      | 同上，但读图片目录，不用摄像头（可多人离线测） |
| `live`       | 跑完整链路，看事件流 —— 端到端自检 |

用法：
    python scripts/bench_perception.py latency
    python scripts/bench_perception.py similarity          # 先注册过才有意义
    python scripts/bench_perception.py pairs --dir data/faces
    python scripts/bench_perception.py live

**阈值必须实测，不能拍脑袋。** `identity.py` 里的 DEFAULT_THRESHOLD = 0.40
只是占位值，跑完 similarity 或 pairs 后按输出改掉它。

数字记进 docs/实验数据.md，并注明条件（模型、det_size、线程数、平台）。
PC 的数字只是参照，10-05 上板后同样的命令要再跑一遍做 PC/板对比。
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.perception import (                               # noqa: E402
    Camera,
    FileIdentityStore,
    PerceptionConfig,
    PerceptionPipeline,
)
from alter_ego.perception.backends.insightface_backend import (  # noqa: E402
    InsightFaceBackend,
)

_log = logging.getLogger("bench")


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[i]


def describe(xs: list[float]) -> str:
    if not xs:
        return "无数据"
    return (f"均值 {statistics.mean(xs):6.2f}  中位 {statistics.median(xs):6.2f}  "
            f"p95 {pct(xs, 95):6.2f}  最大 {max(xs):6.2f}")


# ==================== latency ====================

def cmd_latency(args: argparse.Namespace) -> int:
    """检测与特征提取的单帧耗时。C1 调度器的基线。"""
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.det_sizes.split(",")]
    threads = [int(t) for t in args.threads.split(",")]

    if args.synthetic:
        # 合成帧模式。检测与识别都是固定 shape 的 CNN 前向，
        # 耗时与画面内容无关（只有 NMS 那一小段随人脸数变化，占比极低），
        # 所以不开摄像头也能拿到可用的耗时数字。
        # 代价：画面里没有真脸，检出率和特征提取要走 --synthetic 的替代路径。
        g = np.random.default_rng(0)
        frames = [g.integers(0, 256, (480, 640, 3), dtype=np.uint8)
                  for _ in range(args.frames)]
        print(f"\n合成帧模式：{len(frames)} 帧随机噪声，不开摄像头。")
        print("检测耗时有效（前向计算与内容无关）；特征提取在合成 112×112 "
              "上单独计时。检出率一列无意义。\n")
    else:
        print(f"\n采集 {args.frames} 帧用于测时。请坐在镜头前别动，"
              "让画面里稳定有一张脸 —— 没有脸就测不到特征提取耗时。\n")
        with Camera(args.camera) as cam:
            frames = []
            while len(frames) < args.frames:
                f = cam.read_blocking(timeout=2.0)
                if f is None:
                    print("抓帧超时，摄像头可能被占用")
                    return 1
                frames.append(f)
        print(f"已采集 {len(frames)} 帧，"
              f"尺寸 {frames[0].shape[1]}x{frames[0].shape[0]}\n")

    print(f"{'det_size':>10} {'线程':>4} {'加载s':>6} | 检测耗时 ms (N帧)"
          f"{'':>22} | 特征提取耗时 ms")
    print("-" * 120)

    results = []
    for size in sizes:
        for nt in threads:
            backend = InsightFaceBackend(
                model_name=args.model, det_size=size, num_threads=nt
            )
            det_ms: list[float] = []
            emb_ms: list[float] = []
            n_detected = 0

            # 预热两帧。首次推理含 ORT 的图优化与内存池分配，
            # 混进统计会把均值抬高几十毫秒。
            for f in frames[:2]:
                backend.detect_faces(f)

            for f in frames:
                t0 = time.perf_counter()
                faces = backend.detect_faces(f)
                det_ms.append((time.perf_counter() - t0) * 1000)
                if faces and faces[0].kps is not None:
                    n_detected += 1
                    t1 = time.perf_counter()
                    backend.extract_embedding(f, faces[0])
                    emb_ms.append((time.perf_counter() - t1) * 1000)

            if args.synthetic and not emb_ms:
                # 合成帧里没有真脸，检测器自然不出框。识别模型的输入
                # 是固定的 112×112，耗时与内容无关，所以直接喂合成图计时。
                crop = np.random.default_rng(1).integers(
                    0, 256, (112, 112, 3), dtype=np.uint8
                )
                for _ in range(3):
                    backend._rec.get_feat(crop)          # 预热
                for _ in range(args.frames):
                    t1 = time.perf_counter()
                    backend._rec.get_feat(crop)
                    emb_ms.append((time.perf_counter() - t1) * 1000)

            print(f"{size[0]:>5}x{size[1]:<4} {nt:>4} {backend.load_sec:>6.2f} | "
                  f"{describe(det_ms)} | {describe(emb_ms)}")
            print(f"{'':>10} {'':>4} {'':>6} | 检出率 {n_detected}/{len(frames)}")
            results.append({
                "det_size": f"{size[0]}x{size[1]}", "threads": nt,
                "detect_mean_ms": round(statistics.mean(det_ms), 2),
                "embed_mean_ms": round(statistics.mean(emb_ms), 2) if emb_ms else None,
                "detect_rate": f"{n_detected}/{len(frames)}",
            })

    print("\n写进 docs/实验数据.md 时记上条件：")
    print(f"  模型 {args.model}（det_500m + w600k_mbf），ONNXRuntime CPU，{args.frames} 帧")
    best = min(results, key=lambda r: r["detect_mean_ms"])
    print(f"  最快组合：{best['det_size']} / {best['threads']} 线程 "
          f"→ 检测 {best['detect_mean_ms']} ms")
    print("\n注意：板子 4 核要分给 ASR 与 TTS，感知不该用满线程。"
          "挑线程数时看的是「3 线程够不够快」，不是「几线程最快」。")
    return 0


# ==================== 阈值扫描 ====================

def sweep_threshold(same: list[float], other: list[float]) -> None:
    """扫描阈值，报 FRR（把本人判成陌生人）与 FAR（把别人认成本人）。

    两类错误代价不对称：
      · FRR 高 → 角色把主人当陌生人，尴尬但可挽回（再站近一点就认出来了）
      · FAR 高 → 角色叫错名字、把别人当主人，这是演示现场的灾难
    所以阈值应偏保守，优先压 FAR。
    """
    if not same or not other:
        print("\n样本不足，无法扫描阈值（本人与他人两组都要有数据）")
        return

    print(f"\n本人相似度（{len(same)} 对）: {describe(same)}")
    print(f"  范围 {min(same):.3f} ~ {max(same):.3f}")
    print(f"他人相似度（{len(other)} 对）: {describe(other)}")
    print(f"  范围 {min(other):.3f} ~ {max(other):.3f}")

    gap = min(same) - max(other)
    if gap > 0:
        print(f"\n两组完全分离，间隔 {gap:.3f} —— 阈值取中点 "
              f"{(min(same) + max(other)) / 2:.3f} 即可零错误。")
    else:
        print(f"\n两组有重叠（本人最低 {min(same):.3f} < 他人最高 {max(other):.3f}），"
              "需要在 FAR 与 FRR 之间取舍。")

    print(f"\n{'阈值':>6} {'FRR 漏认本人':>14} {'FAR 错认他人':>14} {'总错误':>8}")
    print("-" * 50)
    rows = []
    for t in [x / 100 for x in range(15, 71, 5)]:
        frr = sum(1 for s in same if s < t) / len(same)
        far = sum(1 for s in other if s >= t) / len(other)
        rows.append((t, frr, far))
        mark = ""
        if far == 0 and frr == 0:
            mark = "  ← 零错误"
        print(f"{t:>6.2f} {frr * 100:>13.1f}% {far * 100:>13.1f}% "
              f"{(frr + far) * 100:>7.1f}%{mark}")

    zero_far = [r for r in rows if r[2] == 0]
    if zero_far:
        pick = min(zero_far, key=lambda r: r[1])
        print(f"\n推荐阈值 {pick[0]:.2f}：FAR 0%（绝不认错人），"
              f"FRR {pick[1] * 100:.1f}%")
    else:
        pick = min(rows, key=lambda r: r[1] + r[2])
        print(f"\n推荐阈值 {pick[0]:.2f}：总错误最低 "
              f"（FRR {pick[1] * 100:.1f}% / FAR {pick[2] * 100:.1f}%）")
        print("没有 FAR=0 的阈值，说明样本采集条件差异过大或档案质量不够，"
              "建议重新注册并增加样本角度。")
    print(f"\n→ 改掉 src/alter_ego/perception/identity.py 的 "
          f"DEFAULT_THRESHOLD = {pick[0]:.2f}")


# ==================== similarity（摄像头，对比已注册档案）====================

def cmd_similarity(args: argparse.Namespace) -> int:
    store = FileIdentityStore(args.data_dir)
    users = store.list_users()
    if not users:
        print("档案库为空。先注册：python scripts/enroll_face.py --name 你的名字")
        return 1
    if not args.me:
        print("已注册用户：")
        for uid, name in users:
            print(f"  {name}")
        print("\n请用 --me <姓名> 指明现在坐在镜头前的是谁。")
        return 1

    me = store.find_by_name(args.me)
    if me is None:
        print(f"档案库里没有「{args.me}」")
        return 1
    others = [store.get(uid) for uid, _ in users if uid != me.user_id]

    print(f"\n现在采 {args.probes} 个探测样本，与档案比对。")
    print(f"镜头前应是「{args.me}」。请在采集过程中自然地变换角度和表情 ——")
    print("这样测出来的本人相似度才覆盖真实使用情况，而不是理想条件。\n")

    backend = InsightFaceBackend(model_name=args.model)
    same: list[float] = []
    other: list[float] = []
    collected = 0

    with Camera(args.camera) as cam:
        while collected < args.probes:
            frame = cam.read_blocking(timeout=2.0)
            if frame is None:
                break
            faces = backend.detect_faces(frame)
            if not faces or faces[0].kps is None:
                continue
            if (faces[0].x2 - faces[0].x1) < args.min_face_px:
                continue

            emb = backend.extract_embedding(frame, faces[0])
            s_me = float(emb @ me.template)
            same.append(s_me)
            for o in others:
                if o is not None:
                    other.append(float(emb @ o.template))

            collected += 1
            ident = store.identify(emb)
            print(f"  [{collected:>2}/{args.probes}] 本人 {s_me:.3f}  "
                  f"判定 {ident.name or '访客':<6} ({ident.similarity:.3f})")
            time.sleep(args.interval)

    if not others:
        print("\n档案库里只有一个人，测不到「他人相似度」。")
        print("两条路补齐：")
        print("  1. 让 z 或 w 也注册一个，再跑一次本命令")
        print("  2. 收集几个人的照片按目录分好，跑 "
              "python scripts/bench_perception.py pairs --dir <目录>")
        print(f"\n当前只有本人相似度：{describe(same)}")
        print(f"  范围 {min(same):.3f} ~ {max(same):.3f}" if same else "")
        return 0

    sweep_threshold(same, other)
    return 0


# ==================== pairs（离线图片目录）====================

def cmd_pairs(args: argparse.Namespace) -> int:
    """读 `<dir>/<人名>/*.jpg` 结构，算全部类内与类间相似度。

    这是最严谨的一组数字 —— 类内对比覆盖同一人的所有样本组合，
    类间覆盖所有跨人组合，比摄像头实时采样的样本量大一个量级。
    """
    import cv2

    root = Path(args.dir)
    if not root.is_dir():
        print(f"目录不存在：{root}")
        print("\n期望结构（每人一个子目录，至少两人、每人至少两张）：")
        print("  data/faces/小明/1.jpg")
        print("  data/faces/小明/2.jpg")
        print("  data/faces/z/1.jpg")
        return 1

    backend = InsightFaceBackend(model_name=args.model)
    per_person: dict[str, list[np.ndarray]] = {}
    skipped: list[str] = []

    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        embs = []
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            # 中文路径下 cv2.imread 会静默返回 None，走 imdecode 绕开
            data = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                skipped.append(f"{img_path} 读不出来")
                continue
            faces = backend.detect_faces(img)
            if not faces or faces[0].kps is None:
                skipped.append(f"{img_path} 没检测到人脸")
                continue
            if len(faces) > 1:
                skipped.append(f"{img_path} 有 {len(faces)} 张脸，取最大的那张")
            embs.append(backend.extract_embedding(img, faces[0]))
        if embs:
            per_person[person_dir.name] = embs
        print(f"  {person_dir.name}: {len(embs)} 张可用")

    for s in skipped:
        print(f"  跳过/注意: {s}")

    if len(per_person) < 2:
        print(f"\n只解析出 {len(per_person)} 个人，算不了类间相似度，至少要两人。")
        return 1

    same: list[float] = []
    other: list[float] = []
    names = list(per_person)
    for i, a in enumerate(names):
        ea = per_person[a]
        for x in range(len(ea)):
            for y in range(x + 1, len(ea)):
                same.append(float(ea[x] @ ea[y]))
        for b in names[i + 1:]:
            for va in ea:
                for vb in per_person[b]:
                    other.append(float(va @ vb))

    print(f"\n{len(names)} 人：{', '.join(names)}")
    sweep_threshold(same, other)
    return 0


# ==================== live（端到端自检）====================

def cmd_live(args: argparse.Namespace) -> int:
    """跑完整链路，打印事件流。确认滞回与身份投票按预期工作。"""
    store = FileIdentityStore(args.data_dir) if not args.detect_only else None
    if store is not None and not store.list_users():
        print("档案库为空，本次只能验证在场判定，认不出具体是谁。")

    backend = InsightFaceBackend(model_name=args.model, num_threads=args.nthreads)
    pipe = PerceptionPipeline(backend, store, PerceptionConfig())

    print(f"\n跑 {args.seconds} 秒。试这几个动作验证滞回是否生效：")
    print("  · 走出画面 3 秒以上 → 应看到 LEFT")
    print("  · 走回来          → 应看到 APPEARED，随后 IDENTIFIED")
    print("  · 短暂低头或转身   → **不应该**出现 LEFT（这就是滞回的作用）")
    print("  · 叫一个人一起进画面 → 应看到 COUNT_CHANGED\n")

    t_end = time.time() + args.seconds
    n_steps = 0
    with Camera(args.camera) as cam:
        while time.time() < t_end:
            frame = cam.read()
            events = pipe.step(frame)
            n_steps += 1
            for ev in events:
                who = ""
                if ev.identity is not None:
                    who = (f"  → {ev.identity.name or '访客'} "
                           f"({ev.identity.similarity:.3f}) {ev.detail}")
                print(f"  [{time.strftime('%H:%M:%S')}] {ev.kind:<14} "
                      f"脸数 {ev.n_faces}{who}")
            time.sleep(max(0.0, 1.0 / pipe.target_fps))

    print(f"\n{n_steps} 次 step，最终状态：")
    for k, v in pipe.stats().items():
        print(f"  {k}: {v}")
    print(f"\n后端耗时统计: {backend.stats.summary()}")
    return 0


# ==================== 入口 ====================

def main() -> int:
    ap = argparse.ArgumentParser(description="感知链路基准测试")
    ap.add_argument("--model", default="buffalo_s")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "users"))
    ap.add_argument("--min-face-px", type=int, default=60)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("latency", help="检测与特征提取耗时扫描")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--det-sizes", default="320x320,640x640")
    p.add_argument("--threads", default="1,2,3,4")
    p.add_argument("--synthetic", action="store_true",
                   help="用合成帧，不开摄像头。耗时数字仍然可用，检出率无意义")
    p.set_defaults(func=cmd_latency)

    p = sub.add_parser("similarity", help="本人/他人相似度与阈值扫描（摄像头）")
    p.add_argument("--me", help="现在坐在镜头前的人的姓名（须已注册）")
    p.add_argument("--probes", type=int, default=30)
    p.add_argument("--interval", type=float, default=0.3)
    p.set_defaults(func=cmd_similarity)

    p = sub.add_parser("pairs", help="本人/他人相似度与阈值扫描（图片目录）")
    p.add_argument("--dir", required=True, help="每人一个子目录")
    p.set_defaults(func=cmd_pairs)

    p = sub.add_parser("live", help="端到端跑链路，看事件流")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--nthreads", type=int, default=3)
    p.add_argument("--detect-only", action="store_true",
                   help="纯检测降级模式，不做身份识别")
    p.set_defaults(func=cmd_live)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中止")
        raise SystemExit(130)
