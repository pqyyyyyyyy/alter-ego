"""SenseVoice 整句识别。录音/VAD 完成后调用 poll 或 transcribe。

模型必须提前放到本地；本模块不下载模型，也不上传原始音频。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import AUDIO_SAMPLE_RATE


class SenseVoiceASR:
    """实现 ASREngine；poll 是整句结束后的识别触发点，无中间结果。"""

    def __init__(self, model_dir: str | Path, num_threads: int = 2):
        model_dir = Path(model_dir)
        model = model_dir / "model.int8.onnx"
        tokens = model_dir / "tokens.txt"
        for path in (model, tokens):
            if not path.is_file():
                raise FileNotFoundError(f"ASR 模型文件缺少：{path}")
        if num_threads < 1:
            raise ValueError("num_threads 必须 >= 1")

        import sherpa_onnx

        self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(tokens), num_threads=num_threads,
            provider="cpu", language="zh", use_itn=True,
        )
        self._buf: list[np.ndarray] = []
        self._samples = 0

    def feed(self, pcm_s16le_16k: bytes) -> None:
        if len(pcm_s16le_16k) % 2:
            raise ValueError("PCM 字节数必须为偶数（16-bit little-endian）")
        if not pcm_s16le_16k:
            return
        count = len(pcm_s16le_16k) // 2
        if self._samples + count > 30 * AUDIO_SAMPLE_RATE:
            raise ValueError("单句音频超过 30 秒，请先 reset 或用 VAD 切句")
        samples = np.frombuffer(pcm_s16le_16k, dtype="<i2")
        self._buf.append(samples.astype(np.float32) / 32768.0)
        self._samples += count

    def transcribe(self) -> str:
        if not self._buf:
            return ""
        audio = np.concatenate(self._buf)
        # 即使推理失败也清空上一句，避免重试时拼接旧音频。
        self.reset()
        stream = self._rec.create_stream()
        stream.accept_waveform(AUDIO_SAMPLE_RATE, audio)
        self._rec.decode_stream(stream)
        return stream.result.text.strip()

    def poll(self) -> tuple[str, bool] | None:
        if not self._buf:
            return None
        return self.transcribe(), True

    def reset(self) -> None:
        self._buf.clear()
        self._samples = 0
