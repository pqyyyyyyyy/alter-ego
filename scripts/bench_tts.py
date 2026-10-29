"""Benchmark the WAV endpoint using the same HTTP client as the board server."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alter_ego.voice.gpt_sovits import RemoteGPTSoVITS


async def benchmark(options):
    client = RemoteGPTSoVITS(options.url, options.timeout)
    output = options.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(options.runs):
        started = time.perf_counter()
        audio = await client.synthesize_wav(options.text)
        elapsed = time.perf_counter() - started
        with wave.open(io.BytesIO(audio), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
            sample_rate = wav.getframerate()
        path = output / f"tts-{index + 1}.wav"
        path.write_bytes(audio)
        result = {
            "run": index + 1,
            "round_trip_seconds": round(elapsed, 3),
            "audio_seconds": round(duration, 3),
            "rtf": round(elapsed / duration, 3),
            "sample_rate": sample_rate,
            "output": str(path.resolve()),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {
        "url": options.url,
        "text": options.text,
        "note": "First run in this benchmark is not necessarily a cold model start. RTF includes HTTP round trip.",
        "results": results,
    }
    (output / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="http://PC-IP:9881/tts")
    parser.add_argument("--text", default="こんにちは。今日は一緒に何をしようか。")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/voice/tts-benchmark")
    options = parser.parse_args()
    if options.runs < 1:
        parser.error("--runs must be at least 1")
    asyncio.run(benchmark(options))


if __name__ == "__main__":
    main()
