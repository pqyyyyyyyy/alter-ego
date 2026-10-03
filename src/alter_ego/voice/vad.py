"""VAD 语音活动检测：判断"有没有人在说话"，不识别内容。

输入：float32、范围 -1~1 的音频数组（sounddevice 录的 int16 必须先 /32768 归一化）
输出：pop_segments() 返回已切分完成的语音片段

参数说明（实测调参对象）：
  threshold            语音检测灵敏度，0.5 起步；太低把环境噪音当语音，太高漏轻声
  min_silence_duration 静音多久算"说完"，桌面陪伴场景推荐 0.6~0.8 秒
  min_speech_duration  太短的语音忽略，防咳嗽/点击声误触
"""
import numpy as np
import sherpa_onnx


class VoiceDetector:
    def __init__(self, model_path: str, sample_rate: int = 16000,
                 threshold: float = 0.35,
                 min_silence_duration: float = 0.6,
                 min_speech_duration: float = 0.25):
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model_path
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence_duration
        cfg.silero_vad.min_speech_duration = min_speech_duration
        # 以下两项必须显式设置，否则 VAD 检测不到任何语音（任务书示例漏了）：
        # window_size 必须与模型输入窗口匹配（silero v5 = 512）
        cfg.silero_vad.window_size = 512
        # 单段最长时长，防止长语音被无限累积
        cfg.silero_vad.max_speech_duration = 20
        cfg.sample_rate = sample_rate
        self._vad = sherpa_onnx.VoiceActivityDetector(
            cfg, buffer_size_in_seconds=30
        )

    def feed(self, samples_f32: np.ndarray) -> None:
        """喂入 float32 格式、范围 -1~1 的音频数组。"""
        self._vad.accept_waveform(samples_f32)

    def pop_segments(self) -> list[np.ndarray]:
        """取出所有已完成的语音段，每段是 float32 数组。"""
        out = []
        while not self._vad.empty():
            out.append(np.array(self._vad.front.samples))
            self._vad.pop()
        return out
