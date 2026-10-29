"""Start the standalone PC text preview without changing the board service."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import uvicorn

from pc.service import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--character", type=Path, default=ROOT / "characters" / "example")
    parser.add_argument("--provider", choices=("dashscope", "deepseek", "zhipu"),
                        default="dashscope")
    parser.add_argument("--local-rewrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    app = create_app(args.character, args.provider, args.local_rewrite)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
