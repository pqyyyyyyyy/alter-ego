"""Create lightweight WebP sidecars for browser-delivered character sprites."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def convert_directory(source: Path, quality: int) -> tuple[int, int, int]:
    converted = 0
    png_bytes = 0
    webp_bytes = 0
    for png in sorted(source.glob("*.png")):
        if png.name.startswith("_"):
            continue
        target = png.with_suffix(".webp")
        with Image.open(png) as image:
            image.save(target, "WEBP", quality=quality, method=6, exact=True)
        converted += 1
        png_bytes += png.stat().st_size
        webp_bytes += target.stat().st_size
    return converted, png_bytes, webp_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Directory containing aligned PNG sprites")
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    if not args.source.is_dir():
        parser.error(f"not a directory: {args.source}")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")

    count, before, after = convert_directory(args.source, args.quality)
    ratio = (after / before * 100) if before else 0
    print(f"converted={count} png_bytes={before} webp_bytes={after} ratio={ratio:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
