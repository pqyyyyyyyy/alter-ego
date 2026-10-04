"""TTS 语音合成：把角色要说的文字转成音频，交给后端播放；同时提供口型时间轴。

严格实现 contracts.TTSEngine 接口（鸭子类型，无需继承）：
  synthesize(text) -> (wav 字节, 采样率)

设计说明：
  - contracts.py 注释明确要求返回 "wav 字节"，所以此处返回带 RIFF 头的完整 wav，
    后端拿到后可直接落盘为 .wav 文件交给前端播放（前端 <audio> 需要文件头）。
  - sid 是说话人编号，aishell3 模型含多音色，可切换（0~9）。
  - 严禁使用真人音色模型（赛事一票否决项），此处仅用开源通用音色。
"""
import io
import wave

import numpy as np
import sherpa_onnx


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1,
                sample_width: int = 2) -> bytes:
    """裸 PCM int16 -> 带 RIFF 头的 wav 字节。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class SherpaTTS:
    def __init__(self, model_dir: str, sid: int = 0, speed: float = 1.0):
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=f"{model_dir}/vits-aishell3.onnx",
                    lexicon=f"{model_dir}/lexicon.txt",
                    tokens=f"{model_dir}/tokens.txt",
                ),
                num_threads=2,
            )
        )
        self._tts = sherpa_onnx.OfflineTts(cfg)
        self._sid = sid
        self._speed = speed

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """返回 (wav 字节, 采样率)。"""
        audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        pcm = (np.array(audio.samples) * 32767).astype(np.int16)
        return _pcm_to_wav(pcm.tobytes(), audio.sample_rate), audio.sample_rate


def mouth_timeline(pcm_int16: np.ndarray, sample_rate: int,
                   hop_ms: int = 20, gain: float = 8.0) -> list[list]:
    """按 20ms 分帧算 RMS 音量，输出 [[时刻ms, 开口度0~1], ...] 给前端做口型。

    gain 实测：aishell3 8kHz 音频下 4.0 偏低（max 0.42），8.0 让开口度多落在 0.2~0.8
    （前端三档量化：<0.33 闭嘴 / 0.33~0.66 半开 / >0.66 全开）。
    """
    hop = int(sample_rate * hop_ms / 1000)
    frames = []
    for i in range(0, len(pcm_int16) - hop, hop):
        chunk = pcm_int16[i:i + hop].astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        frames.append([i * 1000 // sample_rate, min(1.0, rms * gain)])
    return frames
