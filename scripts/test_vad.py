"""VAD 自测：录音 → 归一化 → 分块流式喂给 VoiceDetector → 打印切分结果。

关键修复：sherpa-onnx 的 VoiceActivityDetector 是流式 API，
必须按小块（0.2s）依次 feed，一次性喂整段大音频不会触发切分。
"""
import sounddevice as sd
import numpy as np
import sys

sys.path.insert(0, "src")
from alter_ego.voice.vad import VoiceDetector

SAMPLE_RATE = 16000
CHUNK_SEC = 0.2   # 分块大小，0.2秒一块流式送入


def main():
    vad = VoiceDetector("models/silero_vad.onnx", sample_rate=SAMPLE_RATE)

    print("=== VAD 测试 ===")
    print("将录制 8 秒音频，请说 1~2 句话，中间停顿一下")
    print("倒计时 3 秒…")
    for i in range(3, 0, -1):
        print(f"{i}…")
        sd.sleep(1000)

    print("开始录音…")
    rec = sd.rec(int(8 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print("录音结束，处理中…")

    samples_f32 = rec.astype(np.float32).flatten() / 32768.0
    peak = int(np.abs(rec).max())
    rms = float(np.sqrt(np.mean(samples_f32 ** 2)))
    print(f"录音峰值: {peak}   RMS: {rms:.5f}  {'✅ 有声音' if peak > 1000 else '⚠️ 几乎没录到声音！'}")
    if peak <= 1000:
        print("  → 请确认：倒计时结束后才开始说话、声音正常、离麦克风不要太远")
        return

    # 关键：切成小块流式 feed，不能一次性喂全部
    chunk_size = int(SAMPLE_RATE * CHUNK_SEC)
    for start in range(0, len(samples_f32), chunk_size):
        chunk = samples_f32[start:start + chunk_size]
        vad.feed(chunk)
    # 流结束必须 flush，弹出最后的语音段
    vad._vad.flush()

    segments = vad.pop_segments()

    print(f"\n检测到 {len(segments)} 段语音")
    for i, seg in enumerate(segments):
        duration = len(seg) / SAMPLE_RATE
        print(f"  第 {i + 1} 段：时长 {duration:.2f} 秒，样本数 {len(seg)}")

    if len(segments) > 0:
        print("\n回放第一段语音…")
        play_pcm = (segments[0] * 32768).astype(np.int16)
        sd.play(play_pcm, SAMPLE_RATE)
        sd.wait()
        print("回放完成")
    else:
        print("⚠️ 未检测到语音段：说话声音太小，或 threshold 太高")


if __name__ == "__main__":
    main()
