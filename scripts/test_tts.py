"""TTS 自测：合成语音 → 播放 → 计算口型时间轴 → 统计开口度分布 + RTF。

验收标准：
  1. 能听到合成语音
  2. 口型数组数值合理（开口度 0.2~0.8 占比高，gain 需实测调整）
  3. 记录 TTS RTF
"""
import io
import sys
import time
import wave

import numpy as np
import sounddevice as sd

sys.path.insert(0, "src")
from alter_ego.voice.tts import SherpaTTS, mouth_timeline

TTS_MODEL_DIR = "models/vits-zh-aishell3"


def main():
    print("=== TTS 测试 ===")
    tts = SherpaTTS(TTS_MODEL_DIR, sid=0)
    text = "你好呀，今天天气真不错呢。很高兴见到你。"

    t0 = time.perf_counter()
    wav_bytes, sr = tts.synthesize(text)
    cost = time.perf_counter() - t0

    # 解码 wav 拿回裸 PCM 用于播放和口型
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        sr = w.getframerate()

    dur = len(pcm) / sr
    rtf = cost / dur
    print(f"合成文本: {text}")
    print(f"音频时长 {dur:.2f}s  合成耗时 {cost:.3f}s  RTF = {rtf:.3f}")

    print("\n播放合成语音…")
    sd.play(pcm, sr)
    sd.wait()
    print("播放完成")

    print("\n计算口型时间轴…")
    mouth = mouth_timeline(pcm, sr)
    print(f"总帧数: {len(mouth)}（20ms/帧，共 {len(mouth)*20}ms）")

    vals = [m[1] for m in mouth]
    in_range = sum(1 for v in vals if 0.2 <= v <= 0.8) / len(vals) if vals else 0
    print(f"开口度统计: 最小 {min(vals):.2f}  最大 {max(vals):.2f}  平均 {np.mean(vals):.2f}")
    print(f"0.2~0.8 占比: {in_range:.1%}  {'✅ 合理' if in_range > 0.5 else '⚠️ 需调整 gain'}")
    print(f"前 10 帧: {mouth[:10]}")


if __name__ == "__main__":
    main()
