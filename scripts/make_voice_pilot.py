"""Create a small, private voice-training pilot from reviewed-data drafts.

The output is ignored by git through ``data/`` and is intentionally marked as
an automatic-transcription pilot. It must not be treated as a final dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil


ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def make_pilot(source: Path, output: Path, count: int, seed: int) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录已有内容，避免覆盖：{output}")
    rows = load_rows(source / "review.jsonl")
    candidates = [row for row in rows if row.get("text", "").strip()
                  and not row.get("asr", {}).get("error")]
    if len(candidates) < count:
        raise ValueError(f"可用自动转写片段不足：{len(candidates)} < {count}")

    selected = random.Random(seed).sample(candidates, count)
    audio_dir = output / "samples"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pilot_rows = []
    list_rows = []
    for index, row in enumerate(selected, 1):
        source_audio = source / row["audio"]
        target_name = f"nanami_pilot_{index:02d}.wav"
        target_audio = audio_dir / target_name
        shutil.copy2(source_audio, target_audio)
        text = row["text"].replace("|", "｜").replace("\n", " ").strip()
        pilot_rows.append({
            "id": f"nanami_pilot_{index:02d}",
            "audio": f"samples/{target_name}",
            "source_id": row["id"],
            "text": text,
            "language": "JA",
            "speaker": "nanami",
            "approved": False,
            "text_source": "faster-whisper-small automatic draft",
        })
        list_rows.append(f"{target_audio.as_posix()}|nanami|JA|{text}\n")

    (output / "review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pilot_rows),
        encoding="utf-8",
    )
    (output / "nanami_pilot15.list").write_text("".join(list_rows), encoding="utf-8")
    summary = {
        "count": len(pilot_rows),
        "seed": seed,
        "text_source": "automatic ASR draft; not manually approved",
        "manifest": str((output / "nanami_pilot15.list").resolve()),
        "review": str((output / "review.jsonl").resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="准备私有 15 段语音试验集")
    parser.add_argument("--source", type=Path, default=ROOT / "data/voice/nanami")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/voice/nanami-pilot15")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count 必须 >= 1")
    print(json.dumps(make_pilot(args.source, args.output, args.count, args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
