"""Bounded, model-free PCM segmentation and an async single-worker ASR adapter.
feed/gate/start/stop are owned by the same event loop. Device callbacks must use
loop.call_soon_threadsafe(controller.feed, data). No ALSA/model import here.
"""
from __future__ import annotations
import asyncio
import math
import struct
from collections import deque
from dataclasses import dataclass

@dataclass(frozen=True)
class PCMConfig:
    frame_ms: int = 20
    pre_ms: int = 200
    start_ms: int = 60
    silence_ms: int = 400
    min_ms: int = 100
    max_ms: int = 30000
    rms_threshold: float = 500.0
    def __post_init__(self):
        if self.frame_ms not in (20,30): raise ValueError("frame must be 20/30 ms")
        if not (0 < self.min_ms <= self.max_ms <= 30000): raise ValueError("invalid duration")
        if min(self.pre_ms,self.start_ms,self.silence_ms)<0 or self.start_ms==0 or self.silence_ms==0:
            raise ValueError("invalid boundaries")
        if not math.isfinite(self.rms_threshold) or self.rms_threshold < 0: raise ValueError("invalid threshold")
        if self.pre_ms > self.max_ms or self.start_ms > self.max_ms or self.silence_ms > self.max_ms:
            raise ValueError("boundary exceeds max duration")

class Segmenter:
    sample_rate=16000
    def __init__(self, config=None, vad=None):
        self.cfg=config or PCMConfig(); self.frame_bytes=32*self.cfg.frame_ms
        self.vad=vad or self._energy
        self.gated=False; self.events=deque(maxlen=2048); self.reset()
    def _energy(self, frame):
        samples=struct.unpack('<'+'h'*(len(frame)//2),frame)
        return math.sqrt(sum(x*x for x in samples)/len(samples))>=self.cfg.rms_threshold
    def reset(self):
        self.remainder=bytearray()
        self.pre=deque(maxlen=max(1,math.ceil(self.cfg.pre_ms/self.cfg.frame_ms),
                                 math.ceil(self.cfg.start_ms/self.cfg.frame_ms)))
        self.frames=[]; self.voiced=0; self.hit=0; self.silence=0
    def gate(self, enabled):
        if enabled!=self.gated: self.reset()
        self.gated=enabled
    def feed(self, data):
        if not isinstance(data,bytes): raise TypeError("bytes S16LE required")
        if self.gated: return []
        result=[]
        # Handles odd-byte chunks, holding one byte; bounded processing, no giant-copy buffer.
        view=memoryview(data); pos=0
        while pos<len(view):
            take=min(self.frame_bytes-len(self.remainder),len(view)-pos)
            self.remainder.extend(view[pos:pos+take]); pos+=take
            if len(self.remainder)<self.frame_bytes: break
            frame=bytes(self.remainder); self.remainder.clear()
            try: speech=bool(self.vad(frame))
            except Exception:
                self.reset(); self.events.append('vad_error'); raise
            if not self.frames:
                self.pre.append(frame); self.hit=self.hit+1 if speech else 0
                if self.hit*self.cfg.frame_ms>=self.cfg.start_ms:
                    self.frames=list(self.pre); self.voiced=self.hit; self.silence=0
                continue
            self.frames.append(frame)
            if speech: self.voiced+=1; self.silence=0
            else: self.silence+=1
            if len(self.frames)*self.cfg.frame_ms>=self.cfg.max_ms or self.silence*self.cfg.frame_ms>=self.cfg.silence_ms:
                if self.voiced*self.cfg.frame_ms>=self.cfg.min_ms:
                    pcm=b''.join(self.frames)[:32*self.cfg.max_ms]
                    result.append(pcm)
                    self.events.append('max_length' if len(self.frames)*self.cfg.frame_ms>=self.cfg.max_ms else 'sentence')
                else:self.events.append('short_noise')
                self.frames=[]; self.pre.clear(); self.hit=self.voiced=self.silence=0
        return result
    def end_stream(self):
        """Incomplete sample/frame rejected explicitly; stop does NOT transcribe partial input."""
        incomplete=len(self.remainder)
        self.reset()
        if incomplete: raise ValueError('incomplete PCM frame/sample at EOF')

class InputController:
    def __init__(self, asr, sink, segmenter=None, processor=None):
        self.asr,self.sink=asr,sink; self.segmenter=segmenter or Segmenter()
        self.queue=asyncio.Queue(maxsize=1); self.epoch=0; self.task=None; self.running=False
        self.events=deque(maxlen=2048); self.sequence=0
        self.processor = processor
    async def start(self):
        if self.task and not self.task.done(): return False
        self.running=True; self.task=asyncio.create_task(self._worker()); return True
    def _clear(self):
        while not self.queue.empty(): self.queue.get_nowait(); self.queue.task_done()
    def gate(self, enabled):
        if enabled!=self.segmenter.gated:
            self.epoch+=1; self._clear()
        self.segmenter.gate(enabled)
    def feed(self, data):
        if not self.running: return
        for pcm in self.segmenter.feed(data):
            self.sequence+=1
            try: self.queue.put_nowait((self.epoch,self.sequence,pcm))
            except asyncio.QueueFull:self.events.append('backpressure_drop')
    def _transcribe(self, pcm):
        try:
            self.asr.reset(); self.asr.feed(pcm)
            result = self.asr.poll()
            if isinstance(result, tuple):
                text, final = result
                if final is not True: return ""
                result = text
            if result is not None and not isinstance(result, str):
                raise TypeError("ASR text must be a string")
            return result.strip() if result else ""
        finally:self.asr.reset()
    async def _worker(self):
        while self.running or not self.queue.empty():
            entry=await self.queue.get()
            if entry is None: self.queue.task_done(); break
            epoch,seq,pcm=entry
            try:
                if epoch!=self.epoch or self.segmenter.gated: continue
                if self.processor is not None:
                    await self.processor(pcm, seq)
                    continue
                text=await asyncio.to_thread(self._transcribe,pcm)
                if epoch!=self.epoch or not self.running or self.segmenter.gated:
                    self.events.append('stale_drop'); continue
                if text:
                    # sink is a fast synchronous submission callback; no untracked task.
                    self.sink(text,seq)
                else:self.events.append('empty_asr')
            except Exception as exc:self.events.append('asr_error:'+type(exc).__name__)
            finally:self.queue.task_done()
    async def stop(self):
        if not self.task:return
        if self.running:
            self.running=False; self.epoch+=1; self.segmenter.reset(); self._clear()
            self.queue.put_nowait(None)
        # Cancellation of stop must not cancel the real ASR worker thread.
        await asyncio.shield(self.task)
        self.task=None
