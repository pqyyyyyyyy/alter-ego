"""One reservation owns ASR, dialog/rewrite, synthesis and playback acknowledgement."""
from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
import wave
from dataclasses import asdict

from ..contracts import DialogTurn, PersonaContext
from ..observability import EventLog
from ..persona.output_gate import japanese_error
from ..scheduler.coordinator import Coordinator
from ..sessions import SessionDialogs
from ..voice.input_controller import InputController
from .voice_output import VoiceOutput


async def synthesize(tts, text):
    """Never cancel an adapter that may own a still-running native/thread call."""
    if hasattr(tts, 'synthesize_wav'):
        data = await tts.synthesize_wav(text)
        with wave.open(io.BytesIO(data)) as audio:
            duration = audio.getnframes() / audio.getframerate()
        return 'audio/wav', data, duration
    if hasattr(tts, 'synthesize_mp3'):
        return 'audio/mpeg', await tts.synthesize_mp3(text), None
    data, rate = await asyncio.to_thread(tts.synthesize, text)
    # The legacy TTSEngine contract returns encoded WAV plus its sample rate.
    with wave.open(io.BytesIO(data)) as audio:
        if audio.getframerate() != rate:
            raise ValueError('WAV sample rate mismatch')
        duration = audio.getnframes() / rate
    return 'audio/wav', data, duration


class Lifecycle:
    def __init__(self, runtime, factory, *, model_timeout=120., playback_timeout=240.,
                 stop_timeout=3., baseline=False, clock=time.monotonic, device_id='unspecified'):
        self.rt = runtime
        self.coordinator = Coordinator(model_timeout=model_timeout,
                                       playback_timeout=playback_timeout,
                                       baseline=baseline, clock=clock)
        self.sessions = SessionDialogs(factory)
        self.sessions.arrive()
        self.events = EventLog(device_id, 'server-' + uuid.uuid4().hex)
        self.stop_timeout = stop_timeout
        self.stopping_since = None
        self.tasks = set()
        self.closed = False
        self.watchdog = None
        self.input = None
        self.last_voice = None
        self.status = {name: {'status': 'disabled', 'reason': 'not_enabled'}
                       for name in ('asr', 'tts', 'perception', 'dialog')}
        self.voice_output = VoiceOutput(self, synthesize)

    def track(self, task):
        self.tasks.add(task)
        def finished(done):
            self.tasks.discard(done)
            if not done.cancelled(): done.exception()
        task.add_done_callback(finished)
        return task

    async def start(self):
        self.watchdog = asyncio.create_task(self._watch())

    async def enable_asr(self, asr):
        async def process(pcm, seq):
            self.last_voice = await self.submit('voice', pcm=pcm)
        self.input = InputController(asr, None, processor=process)
        await self.input.start()
        self.status['asr'] = {'status': 'ready', 'reason': 'pcm_only_no_device'}

    async def policy(self, turn=None):
        coord = self.coordinator
        if self.input: self.input.gate(self.voice_output.warming or not coord.budget().voice_submit)
        if turn:
            self.events.add('gate', turn.turn_id, turn.speech_id,
                            enabled=not coord.budget().voice_submit)
        await self.rt.set_state(coord.state)

    async def identify(self, user_id=None, name=None, *, new_guest=False):
        previous = self.sessions.current
        key = 'user:' + str(user_id) if user_id else None
        if key != previous and (user_id or new_guest or not previous or previous.startswith('user:')):
            self.sessions.arrive(user_id, name)
            await self.abort('identity_changed')
        self.rt.current_user = name
        self.rt.current_user_id = user_id

    async def leave(self):
        self.sessions.leave()
        self.rt.current_user = self.rt.current_user_id = None
        self.coordinator.present = False
        await self.abort('left')

    async def abort(self, reason):
        turn = self.coordinator.active
        if not turn: return
        if turn.phase in ('playback', 'stopping') and self.rt.hub.has(turn.owner):
            if turn.phase != 'stopping':
                turn.invalid = reason
                turn.phase = 'stopping'
                self.stopping_since = self.coordinator.clock()
                await self.rt.hub.send({'type': 'stop', 'turn_id': turn.turn_id,
                                        'speech_id': turn.speech_id, 'reason': reason,
                                        'playback_client_id': turn.owner})
        else:
            self.coordinator.abort(reason)
        await self.policy(turn)

    async def disconnect(self, client_id):
        turn = self.coordinator.active
        if turn and turn.owner == client_id:
            if turn.phase == 'stopping':
                self.coordinator._release(turn, 'disconnect')
            else:
                self.coordinator.disconnect(client_id)
            self.events.add('failed', turn.turn_id, turn.speech_id, reason='disconnect')
            await self.policy(turn)

    async def acknowledgement(self, client_id, msg):
        turn = self.coordinator.active
        if (not turn or client_id != turn.owner or msg.get('turn_id') != turn.turn_id or
                msg.get('speech_id') != turn.speech_id): return False
        kind = msg.get('type')
        if kind == 'audio_started' and turn.phase == 'playback':
            if getattr(turn, 'audio_started', False): return False
            turn.audio_started = True
            self.events.add('audio_started', turn.turn_id, turn.speech_id, backend='ws_receipt')
            return True
        result = msg.get('result')
        if kind != 'audio_ended' or result not in ('ended','failed','timeout','cancelled','no_audio'):
            return False
        if turn.phase == 'stopping':
            accepted = self.coordinator._release(turn, turn.invalid or 'stopped')
        else:
            accepted = self.coordinator.acknowledge(client_id, turn.turn_id, turn.speech_id, result)
        if accepted:
            self.events.add('audio_ended', turn.turn_id, turn.speech_id, result=result,
                            backend='ws_receipt')
            await self.policy(turn)
        return accepted

    async def submit(self, source, text='', *, pcm=None, user='', skip_rewrite=False,
                     preview=False):
        if self.closed: return {'ok': False, 'status': 'closed', 'error': 'service_stopping'}
        if self.voice_output.warming:
            return {'ok': False, 'status': 'busy', 'error': 'tts_warming'}
        if pcm is None and (not isinstance(text, str) or not text.strip() or len(text) > 2000):
            return {'ok': False, 'status': 'invalid', 'error': 'text_length'}
        if source == 'say' and self.rt.hub.owner is None:
            return {'ok': False, 'status': 'no_client', 'error': 'no_playback_client'}
        try:
            snapshot = self.sessions.snapshot(user or self.rt.current_user)
        except RuntimeError:
            snapshot = self.sessions.arrive(self.rt.current_user_id, user or self.rt.current_user)
        turn = self.coordinator.reserve(source, self.rt.hub.owner)
        if turn is None:
            self.events.add('rejected', uuid.uuid4().hex, reason='busy')
            return {'ok': False, 'status': 'busy', 'error': 'busy'}
        ctx = PersonaContext(user_name=snapshot.user_name, visual_hint=self.rt.visual_hint)
        voice_enabled = self.voice_output.enabled
        voice_mode = self.voice_output.mode
        self.events.add('turn_start', turn.turn_id, turn.speech_id)
        try:
            await self.policy(turn)
        except asyncio.CancelledError:
            self.coordinator.abort('caller_cancelled')
            raise

        def emit(event): self.events.add(event, turn.turn_id, turn.speech_id)

        async def job():
            nonlocal text
            if pcm is not None:
                emit('vad_end')
                self.events.add('asr_start', turn.turn_id, turn.speech_id, input_audio_s=len(pcm)/32000)
                self.status['asr'] = {'status': 'running', 'reason': ''}
                try:
                    text = await asyncio.to_thread(self.input._transcribe, pcm)
                    self.status['asr'] = {'status': 'ready', 'reason': ''}
                except Exception as exc:
                    self.status['asr'] = {'status': 'failed', 'reason': type(exc).__name__}
                    raise
                finally: emit('asr_end')
                if preview:
                    return {'preview': True, 'text': text or ''}
                if not text: return {'empty': True}
            if turn.invalid or not self.sessions.is_current(snapshot): return {'stale': True}
            if source == 'say':
                if skip_rewrite:
                    reply = DialogTurn(user_text=text, speak_ja=text)
                else:
                    emit('rewrite_start')
                    try:
                        rewritten = await asyncio.to_thread(self.rt.rewriter.rewrite, text, ctx)
                    finally: emit('rewrite_end')
                    reply = DialogTurn(user_text=text, speak_ja=rewritten.text, emotion=rewritten.emotion)
                if japanese_error(reply.speak_ja): raise ValueError('say_requires_japanese')
            else:
                self.status['dialog'] = {'status': 'running', 'reason': ''}
                reply = await asyncio.to_thread(self.sessions.reply, snapshot, text, ctx, emit)
                if reply is None: return {'stale': True}
                self.rt.dialog_error = reply.error
                if reply.error:
                    self.events.add('resources', turn.turn_id, turn.speech_id,
                                    result='degraded', reason='dialog_output')
                self.status['dialog'] = {'status': 'degraded' if reply.error else 'ready',
                                         'reason': 'fallback' if reply.error else ''}
            if turn.invalid or not self.sessions.is_current(snapshot): return {'stale': True}
            audio = None
            preview_sent = False
            if voice_enabled and self.voice_output.enabled and self.rt.tts is not None:
                if voice_mode == 'fast' and self.rt.hub.has(turn.owner):
                    await self.rt.send_emotion(reply.emotion)
                    if turn.invalid or not self.sessions.is_current(snapshot):
                        return {'stale': True}
                    await self.rt.hub.send({'type': 'reply_preview', 'turn_id': turn.turn_id,
                        'speech_id': turn.speech_id, 'playback_client_id': turn.owner,
                        'text': reply.subtitle_zh or reply.speak_ja, 'text_ja': reply.speak_ja})
                    emit('text_sent')
                    preview_sent = True
                emit('tts_start')
                self.status['tts'] = {'status': 'running', 'reason': ''}
                try:
                    mime, data, duration = await synthesize(self.rt.tts, reply.speak_ja)
                    if not data: raise ValueError('empty_audio')
                    audio = 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')
                    self.rt.tts_error = ''
                    self.status['tts'] = {'status': 'ready', 'reason': ''}
                    if duration:
                        self.events.add('resources', turn.turn_id, turn.speech_id, output_audio_s=duration)
                except Exception as exc:
                    self.rt.tts_error = type(exc).__name__
                    self.status['tts'] = {'status': 'failed', 'reason': type(exc).__name__}
                    self.events.add('failed', turn.turn_id, turn.speech_id, reason='tts_error')
                finally: emit('tts_end')
            return {'reply': reply, 'audio': audio if self.voice_output.enabled else None,
                    'preview_sent': preview_sent}

        async def run():
            try:
                if self.coordinator.active is not turn or not self.sessions.is_current(snapshot):
                    self.coordinator._release(turn, 'stale')
                    return {'ok': False, 'status': 'discarded'}
                outcome = await self.coordinator.run(turn, job)
                value = outcome.get('value', {})
                if outcome['status'] in ('failed', 'discarded'):
                    if outcome['status'] == 'failed':
                        self.status['dialog'] = {'status': 'failed', 'reason': outcome['reason']}
                    self.events.add('failed', turn.turn_id, turn.speech_id, reason=outcome['reason'])
                    return {'ok': False, 'status': outcome['status'], 'error': outcome['reason']}
                if value.get('preview'):
                    self.coordinator._release(turn, 'asr_preview')
                    await self.policy(turn)
                    return {'ok': True, 'status': 'asr_preview',
                            'text': value.get('text', ''), 'turn_id': turn.turn_id,
                            'speech_id': turn.speech_id}
                if value.get('empty') or value.get('stale'):
                    self.coordinator._release(turn, 'empty_asr' if value.get('empty') else 'stale')
                    self.events.add('failed', turn.turn_id, turn.speech_id,
                                    reason='empty_asr' if value.get('empty') else 'stale')
                    return {'ok': False, 'status': 'empty' if value.get('empty') else 'discarded'}
                reply = value['reply']
                if outcome['status'] == 'playback':
                    if not self.sessions.is_current(snapshot) or not self.rt.hub.has(turn.owner):
                        await self.abort('stale_owner')
                        return {'ok': False, 'status': 'discarded'}
                    await self.policy(turn)
                    if not value.get('preview_sent'):
                        await self.rt.send_emotion(reply.emotion)
                    if self.coordinator.active is not turn or turn.invalid or not self.sessions.is_current(snapshot):
                        return {'ok': False, 'status': 'discarded'}
                    await self.rt.hub.send({'type': 'speak', 'turn_id': turn.turn_id,
                        'speech_id': turn.speech_id, 'playback_client_id': turn.owner,
                        'text': reply.subtitle_zh or reply.speak_ja, 'text_ja': reply.speak_ja,
                        'audio': value['audio'] if self.voice_output.enabled else None,
                        'tts_error': self.rt.tts_error if voice_enabled else '',
                        'duration_ms': max(400, len(reply.speak_ja)*140)})
                    emit('ws_sent')
                else:
                    self.events.add('failed', turn.turn_id, turn.speech_id, reason='no_client')
                degraded = bool(reply.error or self.rt.tts_error and voice_enabled)
                return {'ok': outcome['status'] == 'playback',
                        'status': 'degraded' if degraded and outcome['status']=='playback' else outcome['status'],
                        'turn_id': turn.turn_id, 'speech_id': turn.speech_id,
                        **asdict(reply), 'tts_error': self.rt.tts_error,
                        'history_turns': snapshot.dialog.history_turns}
            except Exception as exc:
                await self.abort('delivery_error')
                self.events.add('failed', turn.turn_id, turn.speech_id, reason=type(exc).__name__)
                return {'ok': False, 'status': 'failed', 'error': type(exc).__name__}
            finally:
                await self.policy(turn)

        task = self.track(asyncio.create_task(run()))
        try:
            done, _ = await asyncio.wait({task}, timeout=self.coordinator.model_timeout)
            if not done:
                self.events.add('timeout', turn.turn_id, turn.speech_id, reason='model_timeout')
                await self.abort('timeout')
                return {'ok': False, 'status': 'timeout', 'error': 'worker_quarantined',
                        'turn_id': turn.turn_id, 'speech_id': turn.speech_id}
            return task.result()
        except asyncio.CancelledError:
            await self.abort('caller_cancelled')
            raise

    async def tick(self):
        turn = self.coordinator.active
        if not turn: return
        now = self.coordinator.clock()
        if turn.phase == 'stopping':
            if now - self.stopping_since >= self.stop_timeout:
                # Losing stop ACK does not establish silence. Close the owning socket;
                # renderer onclose pauses audio before a fresh connection may play.
                await self.rt.hub.close_client(turn.owner)
                await self.disconnect(turn.owner)
        elif not turn.invalid:
            limit = self.coordinator.playback_timeout if turn.phase == 'playback' else self.coordinator.model_timeout
            if now - turn.started >= limit:
                self.events.add('timeout', turn.turn_id, turn.speech_id, reason=turn.phase+'_timeout')
                await self.abort('timeout')

    async def _watch(self):
        while not self.closed:
            await self.tick()
            await asyncio.sleep(.05)

    async def close(self):
        self.closed = True
        await self.abort('shutdown')
        for client_id in list(self.rt.hub.client_ids):
            await self.rt.hub.close_client(client_id)
            await self.disconnect(client_id)
        if self.watchdog:
            self.watchdog.cancel()
            await asyncio.gather(self.watchdog, return_exceptions=True)
        if self.input: await self.input.stop()
        if self.tasks: await asyncio.gather(*list(self.tasks), return_exceptions=True)
        if self.rt.registry and hasattr(self.rt.registry, 'close'): self.rt.registry.close()

    def health(self):
        turn = self.coordinator.active
        policy = asdict(self.coordinator.budget())
        if self.voice_output.warming: policy['voice_submit'] = False
        return {'protocol': 'aeg-ws-v2', 'device_id': self.events.device_id,
                'clock_id': self.events.clock_id, 'policy': policy,
                'voice_output': self.voice_output.snapshot(),
                'active': {'turn_id': turn.turn_id, 'speech_id': turn.speech_id,
                           'phase': turn.phase, 'invalid': turn.invalid,
                           'age_s': self.coordinator.clock()-turn.started} if turn else None,
                'components': self.status, 'shutdown': self.closed}
