"""音频设备自测：列出设备 → 录 3 秒 → 回放 → 打印峰值音量。

验收标准：
  1. 能录能放
  2. 峰值音量在几千以上（说话时）
  3. 若峰值只有几十或 0，说明麦克风没选对或系统权限没给
"""
import sounddevice as sd
import numpy as np

print("=== 可用设备 ===")
print(sd.query_devices())
print(f"默认输入设备: {sd.default.device[0]}  默认输出设备: {sd.default.device[1]}")

SAMPLE_RATE = 16000
DURATION = 3.0

# 倒计时，给用户准备时间
import time
for c in range(3, 0, -1):
    print(f"倒计时 {c} 秒…", flush=True)
    time.sleep(1)

print(f"\n开始录音 {DURATION:.0f} 秒，请对着麦克风说话…")
rec = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
sd.wait()
print("录音完成，回放…")
sd.play(rec, SAMPLE_RATE)
sd.wait()
print("回放完成")

peak = int(np.abs(rec).max())
print(f"峰值音量: {peak}  {'✅ 正常' if peak > 1000 else '⚠️ 太小，检查麦克风设备选择和系统权限'}")

# 顺便看下均值，帮助判断是否录到了静音
rms = float(np.sqrt(np.mean(rec.astype(np.float32) ** 2 / 32768.0 ** 2)))
print(f"RMS 电平: {rms:.5f}")
