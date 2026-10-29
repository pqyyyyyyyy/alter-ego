"""清点私有语音、准备人工试听队列；不训练、不修改原音频、不联网。

python scripts/prepare_voice_dataset.py --source ../资料引导/语音素材
输出默认位于 gitignore 排除的 data/voice/nanami/。
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import wave


ROOT = Path(__file__).resolve().parents[1]


def inspect_wav(path: Path) -> dict:
    record = {"source": str(path.resolve()), "name": path.name}
    try:
        with wave.open(str(path), "rb") as wav:
            rate, channels, width = (
                wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
            )
            frames = wav.getnframes()
            record.update(sample_rate=rate, channels=channels, sample_width=width,
                          duration_sec=round(frames / rate, 4))
            if channels != 1 or width != 2 or wav.getcomptype() != "NONE":
                raise ValueError("试听筛选仅支持单声道 16-bit PCM WAV")
            pcm = wav.readframes(frames)
            if not frames or len(pcm) != frames * width:
                raise ValueError("音频为空或 WAV 数据不完整")
        values = array("h")
        values.frombytes(pcm)
        if sys.byteorder != "little":
            values.byteswap()
        peak = max(abs(value) for value in values)
        rms = math.sqrt(sum(value * value for value in values) / len(values)) / 32768
        quiet = sum(abs(value) <= 327 for value in values) / len(values)
        clipping = sum(abs(value) >= 32760 for value in values) / len(values)
        record.update(
            pcm_sha256=hashlib.sha256(pcm).hexdigest(),
            peak=peak, rms_dbfs=round(20 * math.log10(max(rms, 1e-12)), 2),
            quiet_fraction=round(quiet, 4), clipping_fraction=round(clipping, 6),
            candidate=(3 <= frames / rate <= 9 and rms >= 0.005
                       and quiet < 0.65 and clipping < 0.001),
            error="",
        )
    except (wave.Error, OSError, ValueError, EOFError, ZeroDivisionError) as exc:
        record.update(candidate=False, error=f"{type(exc).__name__}: {exc}")
    return record


def select_candidates(records: list[dict], count: int) -> list[dict]:
    # 固定随机种子分散抽样，避免连续编号片段都来自同一场景。
    candidates = [record for record in records if record["candidate"]]
    return random.Random(1007).sample(candidates, min(count, len(candidates)))


def prepare(source: Path, output: Path, count: int) -> dict:
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError(f"素材目录不存在：{source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出目录必须与原始素材目录分开")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录已有内容，避免覆盖人工转写：{output}")
    paths = sorted(source.rglob("*.wav"))
    if not paths:
        raise ValueError("没有找到 WAV 文件")
    records = [inspect_wav(path) for path in paths]
    selected = select_candidates(records, count)
    if not selected:
        raise ValueError("没有符合初筛条件的音频；请检查格式、音量和时长")
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    queue = []
    for index, record in enumerate(selected, start=1):
        name = f"nanami_{index:04d}.wav"
        target = samples / name
        shutil.copy2(record["source"], target)
        queue.append(dict(
            record, id=f"nanami_{index:04d}", audio=f"samples/{name}",
            training_path=target.as_posix(), text="", language="JA",
            speaker="nanami", approved=False,
        ))
    (output / "inventory.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (output / "review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue),
        encoding="utf-8",
    )
    template = Path(__file__).with_name("voice_review_template.html").read_text(
        encoding="utf-8"
    )
    # JSON 嵌入 script 时转义 '<'，避免素材文件名被解释为 HTML。
    payload = json.dumps(queue, ensure_ascii=False).replace("<", "\\u003c")
    page = template.replace("__REVIEW_DATA__", payload)
    (output / "review.html").write_text(page, encoding="utf-8")
    summary = {
        "files": len(records),
        "audio_minutes": round(sum(row.get("duration_sec", 0) for row in records) / 60, 2),
        "invalid_files": sum(bool(row["error"]) for row in records),
        "technical_candidates": sum(row["candidate"] for row in records),
        "review_samples": len(queue),
        "review_minutes": round(sum(row["duration_sec"] for row in queue) / 60, 2),
        "review_page": str(output / "review.html"),
        "note": "初筛不验证说话人、语言、背景音或授权；须人工试听和逐句转写",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="私有语音清点和日文转写准备")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/voice/nanami")
    parser.add_argument("--count", type=int, default=48)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count 必须 >= 1")
    print(json.dumps(prepare(args.source, args.output, args.count),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
