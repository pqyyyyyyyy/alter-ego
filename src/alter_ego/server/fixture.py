"""Explicit, model-free fixture. The generated tone is not synthesized speech."""
from __future__ import annotations

import asyncio
import io
import json
import math
import struct
import threading
import wave
from contextlib import contextmanager

from ..dialog import DialogOrchestrator
from ..persona.card import PersonaCard, PersonaProfile
from ..persona.examples import ExamplePool
from ..persona.rewriter import LocalRewriter


class FixtureControl:
    def __init__(self):
        self.response = json.dumps({'ja': '会えてうれしいです。', 'zh': '见到你很高兴。', 'emotion': 'happy'})
        self.cloud_error = self.tts_error = False
        self.asr_result = ('模拟识别结果', True)
        self.releases = {name: threading.Event() for name in ('cloud', 'tts', 'asr')}
        for event in self.releases.values(): event.set()
        self.started = {name: threading.Event() for name in self.releases}
        self.calls = {name: 0 for name in self.releases}
        self.concurrent = self.max_concurrent = 0
        self.messages = []
        self.lock = threading.Lock()
        self.registry = Registry(self)
        self.tts = TTS(self)
        self.asr = ASR(self)

    @contextmanager
    def work(self, stage):
        with self.lock:
            self.calls[stage] += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.started[stage].set()
            self.releases[stage].wait()
            yield
        finally:
            with self.lock: self.concurrent -= 1

    def update(self, data):
        for key in ('response', 'cloud_error', 'tts_error', 'asr_result'):
            if key in data: setattr(self, key, data[key])
        for name in self.releases:
            if data.get('block') == name:
                self.started[name].clear()
                self.releases[name].clear()
            if data.get('release') == name: self.releases[name].set()
        return self.snapshot()

    def snapshot(self):
        return {'calls': self.calls.copy(), 'concurrent': self.concurrent,
                'max_concurrent': self.max_concurrent,
                'started': {name: event.is_set() for name, event in self.started.items()}}


class Registry:
    dialog_chain = ['fake']
    def __init__(self, control): self.control = control
    def chat_with_fallback(self, messages, **kwargs):
        with self.control.work('cloud'):
            self.control.messages.append(messages)
            if self.control.cloud_error: raise ConnectionError('fixture')
            return self.control.response, ''
    def close(self): pass


class TTS:
    def __init__(self, control): self.control = control
    def _wav(self):
        with self.control.work('tts'):
            if self.control.tts_error: raise RuntimeError('fixture_tts_failure')
            out = io.BytesIO()
            with wave.open(out, 'wb') as audio:
                audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                audio.writeframes(struct.pack('<3200h', *[
                    int(1200*math.sin(2*math.pi*440*i/16000)) for i in range(3200)]))
            return out.getvalue()
    async def synthesize_wav(self, text): return await asyncio.to_thread(self._wav)


class ASR:
    def __init__(self, control): self.control = control
    def reset(self): pass
    def feed(self, pcm): pass
    def poll(self):
        with self.control.work('asr'):
            value = self.control.asr_result
            return tuple(value) if isinstance(value, list) else value


def configure(runtime, root):
    runtime.fixture = True
    runtime.fixture_control = FixtureControl()
    runtime.char_dir = None
    runtime.card = PersonaCard(
        name='示例角色', tone='温和', language='ja',
        sentence_endings=['ね'],
        profile=PersonaProfile(fallback_reply='少し待ってね。'),
    )
    runtime.registry = runtime.fixture_control.registry
    runtime.rewriter = LocalRewriter(runtime.card, ExamplePool([]), None)
    runtime.dialog = DialogOrchestrator(runtime.card, runtime.rewriter, runtime.registry)
    runtime.dialog_error = ''
    runtime.tts = runtime.fixture_control.tts
    runtime.tts_error = ''
