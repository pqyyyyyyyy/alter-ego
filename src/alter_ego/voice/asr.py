"""ASR 流式语音识别：把用户说的话转成文字，交给队长模块生成回答。

严格实现 contracts.ASREngine 接口（鸭子类型，无需继承）：
  feed(pcm_s16le_16k: bytes)         喂入 16kHz 单声道 16bit PCM 字节流
  poll() -> (文本, 是否最终结果)|None 取识别结果
  reset()                            一句话结束后必须调用，防止句子拼接

模型：sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20（int8）
  - 中英双语流式 transducer，低延迟
  - int8 量化，CPU 推理 RTF 低（目标 < 0.5）

三个必踩的坑（都对应"识别乱码/空结果"症状）：
  1. 采样率必须 16000（AUDIO_SAMPLE_RATE），录音和重采样都要对齐
  2. int16 必须 /32768 归一化为 float32，漏掉这步识别不出任何东西
  3. poll() 返回 is_final=True 后必须调 reset()，否则下一句拼在上一句后面
"""
import numpy as np
import sherpa_onnx

from ..contracts import AUDIO_SAMPLE_RATE


class SherpaASR:
    def __init__(self, model_dir: str):
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder-epoch-99-avg-1.int8.onnx",
            decoder=f"{model_dir}/decoder-epoch-99-avg-1.int8.onnx",
            joiner=f"{model_dir}/joiner-epoch-99-avg-1.int8.onnx",
            num_threads=2,
            provider="cpu",
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=1.0,
            rule2_min_trailing_silence=0.6,
        )
        self._stream = self._rec.create_stream()

    def feed(self, pcm_s16le_16k: bytes) -> None:
        """喂入 16kHz 单声道 16bit PCM 字节流。"""
        samples = np.frombuffer(pcm_s16le_16k, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768.0  # 必须归一化
        self._stream.accept_waveform(AUDIO_SAMPLE_RATE, samples)

    def poll(self) -> tuple[str, bool] | None:
        """返回 (文本, 是否最终结果)；无结果时返回 None。"""
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        text = self._rec.get_result(self._stream)
        is_final = self._rec.is_endpoint(self._stream)
        if not text and not is_final:
            return None
        return (text, is_final)

    def reset(self) -> None:
        """一句话结束后必须调用，否则下一句会接在上一句后面。"""
        self._rec.reset(self._stream)
