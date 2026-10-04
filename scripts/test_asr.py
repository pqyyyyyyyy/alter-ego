"""ASR 自测：录音 → VAD 切段 → ASR 流式识别 → 打印文本 + RTF。

验收标准：
  1. 中文识别文本正确
  2. RTF < 0.5（必须 < 1）
"""
import sounddevice as sd
import numpy as np
import sys
import time

sys.path.insert(0, "src")
from alter_ego.voice.vad import VoiceDetector
from alter_ego.voice.asr import SherpaASR

SR = 16000
ASR_MODEL_DIR = "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"


def main():
    print("=== ASR 测试 ===")
    print("请准备说一句话（如：今天天气怎么样）")
    for i in range(3, 0, -1):
        print(f"{i}…")
        sd.sleep(1000)

    print("开始录音（6 秒）…")
    rec = sd.rec(int(6 * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    print("录音完成")

    peak = int(np.abs(rec).max())
    print(f"录音峰值: {peak}  {'✅ 有声音' if peak > 1000 else '⚠️ 几乎没录到声音'}")
    if peak <= 1000:
        return

    # 1. VAD 切出语音段
    print("\n[1/3] VAD 切段…")
    vad = VoiceDetector("models/silero_vad.onnx", sample_rate=SR)
    f32 = rec.astype(np.float32).flatten() / 32768.0
    chunk = int(SR * 0.2)
    for i in range(0, len(f32), chunk):
        vad.feed(f32[i:i + chunk])
    vad._vad.flush()
    segs = vad.pop_segments()
    print(f"切出 {len(segs)} 段")

    if not segs:
        print("⚠️ 未检测到语音，请大声一点重试")
        return

    # 2. 每段喂 ASR
    print("\n[2/3] ASR 识别…")
    asr = SherpaASR(ASR_MODEL_DIR)
    total_rtf = 0.0
    for i, seg in enumerate(segs):
        pcm16 = (seg * 32767).astype(np.int16)
        pcm_bytes = pcm16.tobytes()
        dur = len(pcm16) / SR

        t0 = time.perf_counter()
        # 段尾补 0.8s 静音，帮助 endpoint 判定句子结束
        pad = np.zeros(int(0.8 * SR), dtype=np.int16)
        pcm16_padded = np.concatenate([pcm16, pad])
        asr.feed(pcm16_padded.tobytes())
        text = ""
        # 有限轮询：拿到中间结果即记录，最多 10 次（防 endpoint 不触发导致死循环）
        for _ in range(10):
            r = asr.poll()
            if r is None:
                break
            text = r[0]
            if r[1]:
                break
        cost = time.perf_counter() - t0

        rtf = cost / dur
        total_rtf += rtf
        print(f"  段{i+1}: 「{text}」 时长{dur:.2f}s 耗时{cost:.3f}s RTF={rtf:.3f}")
        asr.reset()

    print(f"\n[3/3] 平均 RTF = {total_rtf/len(segs):.3f}  {'✅ < 0.5 达标' if total_rtf/len(segs) < 0.5 else '⚠️ 偏慢，需优化'}")


if __name__ == "__main__":
    main()
