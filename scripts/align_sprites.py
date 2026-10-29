"""立绘对齐：把各自裁过的立绘统一到同一画布、同一头部位置。

**为什么必须做这步**

游戏素材被提取时每张都裁到了自己的 alpha 外接框，画布尺寸各不相同
（实测 17 张有 15 种尺寸），头部水平位置极差 110 px。直接用的后果是
切换表情时头会跳一下 —— 这个瑕疵不报错，但录演示视频时非常显眼。

**锚点为什么不用人脸检测**

试过了：SCRFD 是真人脸训练的，17 张里有 6 张检不到动漫脸（侧脸、闭眼、
手遮）。改用**头部轮廓模板匹配** —— 同一角色同一发型，发型轮廓跨姿势
几乎一致（实测发区宽比值 1.11），比五官稳得多。

**为什么只平移不缩放**

同一套游戏素材，人物本体是同尺寸的。早先量到的「头宽 1.53 倍差」是
举手动作伸进头部区域造成的假象，不是真的缩放差异。盲目加缩放只会引入误差。

用法：
    python scripts/align_sprites.py --character nanami
    python scripts/align_sprites.py --character nanami --ref usual --check-only

输出：
    sprites/aligned/<原名>.png   统一画布的 RGBA
    sprites/aligned/_check_overlay.png    半透明叠加图，头不跳才算对
    sprites/aligned/_check_grid.png       对齐后的九宫格
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 模板取内容顶部这么多行。200 行覆盖头发与额头，
# 不会低到胸前 —— 多数姿势的手停在下巴以下，纳进来会污染匹配。
HEAD_ROWS = 200

# 搜索余量。发顶质心极差实测 110 px，±200 够用且不会慢。
PAD = 220


def load_rgba(path: Path) -> np.ndarray:
    """按 RGBA 读图。中文路径下 cv2.imread 会静默返回 None，走 imdecode。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"读不出来: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        # 没有 alpha 的图不该进来 —— 补一个全不透明的 alpha 会让它
        # 变成一块实心矩形盖住背景，而且不报错。这里显式拒绝。
        raise ValueError(f"{path.name} 没有 alpha 通道，不能用作立绘")
    return img


def content_bbox(alpha: np.ndarray, thr: int = 10) -> tuple[int, int, int, int]:
    m = alpha > thr
    ys = np.where(m.any(axis=1))[0]
    xs = np.where(m.any(axis=0))[0]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def head_template(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """取参照图的头部轮廓作模板。返回 (模板, 模板在原图的 x, y)。"""
    a = img[:, :, 3]
    x0, y0, x1, _ = content_bbox(a)
    y1 = min(a.shape[0], y0 + HEAD_ROWS)
    tpl = (a[y0:y1, x0:x1 + 1] > 10).astype(np.float32)
    return tpl, x0, y0


def find_offset(img: np.ndarray, tpl: np.ndarray, tx: int, ty: int) -> tuple[int, int, float]:
    """模板匹配求该图相对参照图的平移量。返回 (dx, dy, 匹配分)。

    dx/dy 是「把本图平移多少能对上参照图」。
    """
    a = (img[:, :, 3] > 10).astype(np.float32)
    padded = cv2.copyMakeBorder(a, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0.0)
    if padded.shape[0] < tpl.shape[0] or padded.shape[1] < tpl.shape[1]:
        return 0, 0, 0.0
    res = cv2.matchTemplate(padded, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    # loc 是模板左上角在 padded 里的位置，换回未 pad 的坐标
    mx, my = loc[0] - PAD, loc[1] - PAD
    # 本图的头在 (mx,my)，参照图的头在 (tx,ty)，所以要平移这个差
    return tx - mx, ty - my, float(score)


def main() -> int:
    ap = argparse.ArgumentParser(description="立绘对齐")
    ap.add_argument("--character", default="nanami")
    ap.add_argument("--src", default="sprites/selected", help="相对角色目录")
    ap.add_argument("--out", default="sprites/aligned", help="相对角色目录")
    ap.add_argument("--ref", default="usual",
                    help="参照图名（不含扩展名）。应选姿势最标准的一张")
    ap.add_argument("--margin", type=int, default=12, help="画布四周留白")
    ap.add_argument("--check-only", action="store_true", help="只报偏移量，不写文件")
    args = ap.parse_args()

    char_dir = ROOT / "characters" / args.character
    src_dir = char_dir / args.src
    out_dir = char_dir / args.out
    if not src_dir.is_dir():
        print(f"源目录不存在: {src_dir}")
        return 1

    paths = sorted(p for p in src_dir.iterdir()
                   if p.suffix.lower() == ".png" and not p.name.startswith("_"))
    if not paths:
        print(f"{src_dir} 下没有 png")
        return 1

    ref_path = src_dir / f"{args.ref}.png"
    if not ref_path.exists():
        print(f"参照图不存在: {ref_path}")
        print("可选:", ", ".join(p.stem for p in paths))
        return 1

    ref = load_rgba(ref_path)
    tpl, tx, ty = head_template(ref)
    print(f"参照图 {args.ref}  画布 {ref.shape[1]}x{ref.shape[0]}  "
          f"模板 {tpl.shape[1]}x{tpl.shape[0]} @ ({tx},{ty})\n")

    # 第一遍：求偏移，并算出平移后所有图的并集边界
    items = []
    print(f"{'文件':<20}{'原画布':>12}{'dx':>6}{'dy':>6}{'匹配分':>8}")
    print("-" * 54)
    for p in paths:
        img = load_rgba(p)
        if p.stem == args.ref:
            dx, dy, sc = 0, 0, 1.0
        else:
            dx, dy, sc = find_offset(img, tpl, tx, ty)
        x0, y0, x1, y1 = content_bbox(img[:, :, 3])
        items.append({
            "name": p.stem, "img": img, "dx": dx, "dy": dy, "score": sc,
            "bbox": (x0, y0, x1, y1),
        })
        flag = "" if sc > 0.80 else "   ← 匹配分低，务必人工核对"
        print(f"{p.stem:<20}{f'{img.shape[1]}x{img.shape[0]}':>12}"
              f"{dx:>6}{dy:>6}{sc:>8.3f}{flag}")

    # 并集边界（平移后的坐标系）
    gx0 = min(it["bbox"][0] + it["dx"] for it in items)
    gy0 = min(it["bbox"][1] + it["dy"] for it in items)
    gx1 = max(it["bbox"][2] + it["dx"] for it in items)
    gy1 = max(it["bbox"][3] + it["dy"] for it in items)
    W = gx1 - gx0 + 1 + args.margin * 2
    H = gy1 - gy0 + 1 + args.margin * 2
    print(f"\n统一画布 {W}x{H}（并集边界 + 留白 {args.margin}）")

    low = [it["name"] for it in items if it["score"] <= 0.80]
    if low:
        print(f"匹配分偏低的 {len(low)} 张需人工核对: {', '.join(low)}")

    if args.check_only:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    canvases = {}
    for it in items:
        img = it["img"]
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        # 本图像素 (x,y) → 画布 (x + dx - gx0 + margin, y + dy - gy0 + margin)
        ox = it["dx"] - gx0 + args.margin
        oy = it["dy"] - gy0 + args.margin
        h, w = img.shape[:2]
        # 目标与源的交集，避免越界
        dx0, dy0 = max(0, ox), max(0, oy)
        dx1, dy1 = min(W, ox + w), min(H, oy + h)
        sx0, sy0 = dx0 - ox, dy0 - oy
        canvas[dy0:dy1, dx0:dx1] = img[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        cv2.imencode(".png", canvas)[1].tofile(str(out_dir / f"{it['name']}.png"))
        canvases[it["name"]] = canvas

    print(f"\n已写入 {len(canvases)} 张到 {out_dir}")
    _write_checks(canvases, out_dir, args.ref)
    print("\n**务必打开这两张图确认**：")
    print(f"  {out_dir / '_check_overlay.png'}   所有图半透明叠加，"
          "轮廓重合说明对齐成功；头部出现重影就是没对上")
    print(f"  {out_dir / '_check_grid.png'}      对齐后逐张预览")
    return 0


def _write_checks(canvases: dict[str, np.ndarray], out_dir: Path, ref: str) -> None:
    """叠加图与网格图。对齐对不对只能靠眼睛最终确认。"""
    names = list(canvases)
    H, W = canvases[names[0]].shape[:2]

    # 叠加：每张取边缘后累加，重合则线条重叠，错位则出现重影
    acc = np.zeros((H, W), dtype=np.float32)
    for nm in names:
        a = canvases[nm][:, :, 3]
        edge = cv2.Canny((a > 10).astype(np.uint8) * 255, 50, 150)
        acc += (edge > 0).astype(np.float32)
    acc = acc / max(1.0, acc.max())
    heat = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # 参照图压在底下做对比
    ra = canvases[ref][:, :, 3:4] / 255.0
    base = (canvases[ref][:, :, :3] * ra + 255 * (1 - ra)).astype(np.uint8)
    overlay = cv2.addWeighted(base, 0.45, heat, 0.55, 0)
    cv2.imencode(".png", overlay)[1].tofile(str(out_dir / "_check_overlay.png"))

    # 网格
    CW, CH = 240, 320
    cells = []
    for nm in names:
        img = canvases[nm]
        a = img[:, :, 3:4] / 255.0
        comp = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        comp = cv2.resize(comp, (CW, CH - 28), interpolation=cv2.INTER_AREA)
        cell = np.full((CH, CW, 3), 255, np.uint8)
        cell[28:, :] = comp
        cv2.putText(cell, nm[:24], (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 200), 1)
        cv2.rectangle(cell, (0, 0), (CW - 1, CH - 1), (160, 160, 160), 1)
        # 中线：对齐后所有图的头应压在同一条线上
        cv2.line(cell, (CW // 2, 28), (CW // 2, CH - 1), (0, 200, 0), 1)
        cells.append(cell)
    while len(cells) % 6:
        cells.append(np.full((CH, CW, 3), 255, np.uint8))
    rows = [np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)]
    cv2.imencode(".png", np.vstack(rows))[1].tofile(str(out_dir / "_check_grid.png"))


if __name__ == "__main__":
    raise SystemExit(main())
