"""Shared voice switch and one real, non-cancelled warmup per app session."""
from __future__ import annotations

import asyncio


class VoiceOutput:
    def __init__(self, lifecycle, synthesizer):
        self.lifecycle = lifecycle
        self.rt = lifecycle.rt
        self.synthesizer = synthesizer
        self.enabled = bool(getattr(self.rt, 'fixture', False) and self.rt.tts is not None)
        self.mode = 'fast'
        self.warmed = self.enabled
        self.warming = False
        self.error = ''
        self.lock = asyncio.Lock()

    def snapshot(self):
        return {'configured': self.rt.tts is not None, 'enabled': self.enabled,
                'mode': self.mode, 'status': 'warming' if self.warming else
                'failed' if self.error else 'ready' if self.enabled and self.warmed else 'off',
                'error': self.error}

    async def publish(self):
        await self.rt.hub.send({'type': 'voice_output', **self.snapshot()})
        await self.lifecycle.policy()

    async def control(self, enabled, mode):
        async with self.lock:
            if self.lifecycle.closed:
                return {'ok': False, 'error': 'service_stopping', 'voice': self.snapshot()}
            # OFF may mute an active turn; native synthesis is never cancelled.
            if enabled and (self.lifecycle.coordinator.active or self.warming):
                return {'ok': False, 'error': 'busy', 'voice': self.snapshot()}
            if enabled and self.rt.tts is None:
                return {'ok': False, 'error': 'tts_not_configured', 'voice': self.snapshot()}
            self.enabled, self.mode, self.error = enabled, mode, ''
            if enabled and not self.warmed:
                self.warming = True
                self.lifecycle.status['tts'] = {'status': 'loading', 'reason': 'warmup'}
                # Publish the gate before the task can complete and publish READY.
                await self.publish()
                self.lifecycle.track(asyncio.create_task(self._warmup()))
            else:
                await self.publish()
            return {'ok': True, 'voice': self.snapshot()}

    async def _warmup(self):
        try:
            if hasattr(self.rt.tts, 'check_ready'):
                await self.rt.tts.check_ready()
            _, data, _ = await self.synthesizer(self.rt.tts, 'おはよう。')
            if not data:
                raise ValueError('empty_warmup_audio')
            self.warmed = True
            self.rt.tts_error = ''
            self.lifecycle.status['tts'] = {'status': 'ready', 'reason': ''}
        except Exception as exc:
            self.error = type(exc).__name__
            self.enabled = False
            self.rt.tts_error = self.error
            self.lifecycle.status['tts'] = {'status': 'failed', 'reason': self.error}
        finally:
            self.warming = False
            await self.publish()
