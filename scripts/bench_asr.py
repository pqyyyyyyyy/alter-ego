"""板端 ASR 自测：读取 WAV 或显式录音，可选送入已有网页对话。

python scripts/bench_asr.py --wav data/audio-test.wav
python scripts/bench_asr.py --record 6 --device plughw:CARD=Audio,DEV=0
python scripts/bench_asr.py --record 6 --send-to-server

录音用 Linux ALSA arecord，模型用已安装的 sherpa-onnx；不依赖 sounddevice。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alter_ego.contracts import AUDIO_SAMPLE_RATE  # noqa: E402
from alter_ego.voice.asr import SenseVoiceASR  # noqa: E402

MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
            AUDIO_SAMPLE_RATE, 1, 2
        ) or wav.getcomptype() != "NONE":
            raise ValueError("WAV 必须为 16000 Hz、单声道、16-bit PCM；请重新录制")
        frames = wav.getnframes()
        if not 0 < frames <= 30 * AUDIO_SAMPLE_RATE:
            raise ValueError("WAV 时长必须在 0～30 秒之间")
        pcm = wav.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError("WAV 数据不完整")
        return pcm


def record_pcm(seconds: int, device: str) -> bytes:
    if not shutil.which("arecord"):
        raise RuntimeError("未找到 ALSA arecord；也可改用 --wav 读取录音文件")
    print(f"现在开始录音 {seconds} 秒，请说：你好，今天一起做什么？", flush=True)
    with tempfile.TemporaryDirectory(prefix="alter-ego-asr-") as directory:
        path = Path(directory) / "recording.wav"
        # plughw 由 ALSA 转为固定 16kHz 单声道，不假设 USB 硬件直接支持此格式。
        subprocess.run([
            "arecord", "-D", device, "-t", "wav", "-f", "S16_LE",
            "-r", str(AUDIO_SAMPLE_RATE), "-c", "1", "-d", str(seconds),
            str(path),
        ], check=True, timeout=seconds + 10)
        return read_pcm(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="本地中文 ASR 与网页对话自测")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path, help="16kHz 单声道 16-bit WAV")
    source.add_argument("--record", type=int, choices=range(1, 31), metavar="秒",
                        help="从板端麦克风录制 1～30 秒，不保留录音")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / MODEL_NAME)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="plughw:CARD=Audio,DEV=0")
    parser.add_argument("--send-to-server", action="store_true",
                        help="把识别文字发到本机 /debug/chat，浏览器应保持打开")
    args = parser.parse_args()

    start = time.perf_counter()
    asr = SenseVoiceASR(args.model_dir, num_threads=args.threads)
    load_sec = time.perf_counter() - start
    pcm = read_pcm(args.wav) if args.wav else record_pcm(args.record, args.device)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    peak = int(np.max(np.abs(samples)))
    duration = len(samples) / AUDIO_SAMPLE_RATE
    rms = float(np.sqrt(np.mean((samples / 32768.0) ** 2)))
    print(f"录音 {duration:.2f}s，峰值 {peak}，RMS {rms:.5f}", flush=True)
    if peak == 0:
        raise ValueError("录音全为静音，请检查麦克风设备/静音开关")

    start = time.perf_counter()
    asr.feed(pcm)
    text = asr.transcribe()
    elapsed = time.perf_counter() - start
    print(json.dumps({
        "text": text, "load_sec": round(load_sec, 3),
        "audio_sec": round(duration, 3), "asr_sec": round(elapsed, 3),
        "rtf": round(elapsed / duration, 3), "threads": args.threads,
        "provider": "cpu", "timing": "首次识别，未预热",
    }, ensure_ascii=False), flush=True)
    if not text:
        print("未识别到文字，本次不发送对话。", file=sys.stderr)
        return 1

    if args.send_to_server:
        import requests

        health = requests.get("http://127.0.0.1:8000/health", timeout=10)
        health.raise_for_status()
        if health.json().get("clients", 0) < 1:
            raise RuntimeError("请先在电脑打开角色网页，再测试语音对话")
        response = requests.post(
            "http://127.0.0.1:8000/debug/chat", params={"text": text}, timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, ensure_ascii=False))
        if not result.get("ok") or result.get("error") or result.get("tts_error"):
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"测试失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
