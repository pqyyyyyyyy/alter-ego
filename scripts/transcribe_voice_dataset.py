"""本机日文转写初稿；支持断点续跑，不自动批准训练片段。

使用独立 tools/voice-labeling/.venv，模型必须预先下载到本机。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import wave

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def model_identity(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("model.bin", "config.json", "tokenizer.json"):
        with (path / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def merge_draft(row: dict, draft: dict) -> dict:
    merged = dict(row)
    if draft.get("error"):
        return merged
    edited = row.get("text_edited", bool(row.get("text", "").strip()
                                        and not row.get("asr")))
    if not edited and not row.get("approved"):
        merged["text"] = draft["text"]
        merged["text_edited"] = False
    merged["asr"] = draft
    return merged


def publish(directory: Path, rows: list[dict]) -> None:
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("voice_review_template.html").read_text(
        encoding="utf-8")
    for name, content in (
        ("review.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in rows)),
        ("review.html", template.replace("__REVIEW_DATA__", payload)),
    ):
        target = directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)


def transcribe(directory: Path, model_path: Path, limit: int, threads: int) -> None:
    from faster_whisper import WhisperModel

    rows = load_rows(directory / "review.jsonl")
    if not rows:
        raise ValueError("review.jsonl 为空或不存在")
    identity = model_identity(model_path)
    journal = directory / "asr-drafts.jsonl"
    cached = {(d["pcm_sha256"], d["model_sha256"]): d
              for d in load_rows(journal) if not d.get("error")}
    model = None
    completed = 0
    selected = rows[:limit] if limit else rows
    for index, row in enumerate(selected):
        audio = (directory / row["audio"]).resolve()
        if directory.resolve() not in audio.parents:
            raise ValueError("试听音频必须位于数据目录内")
        with wave.open(str(audio), "rb") as wav:
            actual_hash = hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest()
        if actual_hash != row["pcm_sha256"]:
            raise ValueError(f"音频与清单哈希不匹配：{audio.name}")
        key = (actual_hash, identity)
        draft = cached.get(key)
        if draft is None:
            if model is None:
                print("加载本地模型，CPU int8…", flush=True)
                model = WhisperModel(str(model_path), device="cpu", compute_type="int8",
                                     cpu_threads=threads, local_files_only=True)
            started = time.perf_counter()
            draft = dict(pcm_sha256=actual_hash, model_sha256=identity,
                         model_path=str(model_path.resolve()), language="ja",
                         text="", warnings=[], error="")
            try:
                segments, _ = model.transcribe(
                    str(audio), language="ja", task="transcribe", beam_size=5,
                    condition_on_previous_text=False, temperature=0, vad_filter=True)
                segments = list(segments)
                draft["text"] = "".join(s.text for s in segments).strip()
                draft["segments"] = [dict(start=s.start, end=s.end, text=s.text,
                    avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob)
                    for s in segments]
                if not draft["text"]:
                    draft["warnings"].append("未识别出台词")
                if any(s.avg_logprob < -1 for s in segments):
                    draft["warnings"].append("识别不确定，请重点试听")
                if any(s.no_speech_prob > 0.5 for s in segments):
                    draft["warnings"].append("可能含非对白")
            except Exception as exc:
                draft["error"] = f"{type(exc).__name__}: {exc}"
            draft["elapsed_sec"] = round(time.perf_counter() - started, 3)
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(draft, ensure_ascii=False) + "\n")
        rows[index] = merge_draft(row, draft)
        publish(directory, rows)
        completed += not bool(draft["error"])
        print(f"[{index + 1}/{len(selected)}] {row['id']} "
              f"{draft.get('elapsed_sec', 0)}s: {draft['error'] or draft['text']}",
              flush=True)
    print(f"完成 {completed}/{len(selected)} 段初稿，全部仍需试听核对。", flush=True)
    if completed != len(selected):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 Whisper 日文转写初稿")
    parser.add_argument("--directory", type=Path, default=ROOT / "data/voice/nanami")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 表示全部")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1:
        parser.error("limit 必须 >= 0，threads 必须 >= 1")
    transcribe(args.directory, args.model, args.limit, args.threads)


if __name__ == "__main__":
    main()
