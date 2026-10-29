"""Single-turn coordinator. A timed-out worker remains quarantined until done.
Owner identity MUST be bound to the websocket by the server, not user JSON.
"""
from __future__ import annotations
import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Budget:
    voice_submit: bool
    identity_extract: bool
    presence_fps: float
    policy: str

@dataclass
class Turn:
    turn_id: str
    speech_id: str
    source: str
    owner: str | None
    started: float
    phase: str = "model"
    invalid: str = ""
    worker: asyncio.Task | None = None

class Coordinator:
    version = "aeg-sched-v1"
    queue_limit = 0  # explicit reject policy; <= 1, never queue microphone echo

    def __init__(self, *, clock: Callable = time.monotonic, baseline=False,
                 model_timeout=30.0, playback_timeout=120.0, presence_fps=1.0):
        if model_timeout <= 0 or playback_timeout <= 0 or presence_fps <= 0:
            raise ValueError("positive timeout/budget required")
        self.clock, self.baseline = clock, baseline
        self.model_timeout, self.playback_timeout = model_timeout, playback_timeout
        self.presence_fps = presence_fps
        self.active = None
        self.present = False
        self.events = deque(maxlen=2048)

    def _event(self, event, turn=None, **kw):
        self.events.append(dict(event=event, mono_s=self.clock(), policy=self.version,
                                turn_id=turn.turn_id if turn else None, **kw))

    @property
    def state(self):
        if not self.active: return "NOTICED" if self.present else "IDLE"
        return "SPEAKING" if self.active.phase in ("playback", "stopping") else "DIALOG"

    def budget(self):
        busy = self.active is not None
        return Budget(not busy, self.baseline or not busy,
                      5.0 if self.baseline or not busy else self.presence_fps,
                      "baseline" if self.baseline else self.version)

    def reserve(self, source, owner=None):
        if source not in ("voice", "chat", "say"): raise ValueError("unknown source")
        if self.active:
            self._event("rejected", self.active, source=source, reason="busy")
            return None
        turn = Turn(uuid.uuid4().hex, uuid.uuid4().hex, source, owner, self.clock())
        self.active = turn
        self._event("accepted", turn, source=source)
        return turn

    def _release(self, turn, reason):
        if self.active is not turn: return False
        self.active = None
        self._event("released", turn, reason=reason)
        return True

    def abort(self, reason):
        turn = self.active
        if not turn: return False
        if turn.phase == "playback" or turn.worker is None:
            return self._release(turn, reason)
        turn.invalid = turn.invalid or reason
        turn.phase = "quarantine"
        self._event("quarantine", turn, reason=reason)
        return True

    def expire(self):
        turn = self.active
        if not turn: return False
        limit = self.playback_timeout if turn.phase == "playback" else self.model_timeout
        if turn.invalid: return False
        if self.clock() - turn.started >= limit: return self.abort("timeout")
        return False

    def disconnect(self, client_id):
        if self.active and self.active.owner == client_id: return self.abort("disconnect")
        return False

    def leave(self):
        self.present = False
        return self.abort("left")

    async def run(self, turn, job):
        """job is async; wrap synchronous inference in asyncio.to_thread INSIDE it.
        Caller cancellation does not stop that thread; callback holds the reservation.
        The adapter must await this entire job, not spawn untracked inference.
        """
        if self.active is not turn or turn.worker is not None:
            raise ValueError("invalid/already running reservation")
        turn.worker = asyncio.create_task(job())
        def finished(worker):
            # Always consume exceptions, including late exceptions after cancellation.
            if not worker.cancelled(): worker.exception()
            if turn.invalid: self._release(turn, turn.invalid)
        turn.worker.add_done_callback(finished)
        try:
            value = await asyncio.shield(turn.worker)
        except asyncio.CancelledError:
            self.abort("caller_cancelled")
            raise
        except Exception as exc:
            self._release(turn, "model_error")
            return {"status": "failed", "reason": type(exc).__name__}
        if turn.invalid or self.active is not turn:
            self._release(turn, turn.invalid or "stale")
            return {"status": "discarded", "reason": turn.invalid or "stale"}
        if turn.owner is None:
            self._release(turn, "no_client")
            return {"status": "no_client", "value": value}
        turn.phase, turn.started = "playback", self.clock()
        self._event("playback_wait", turn)
        return {"status": "playback", "value": value,
                "turn_id": turn.turn_id, "speech_id": turn.speech_id}

    def acknowledge(self, client_id, turn_id, speech_id, result="ended"):
        turn = self.active
        if (not turn or turn.phase != "playback" or client_id != turn.owner or
            turn_id != turn.turn_id or speech_id != turn.speech_id or
            result not in ("ended", "failed", "timeout", "cancelled", "no_audio")):
            self._event("invalid_ack", turn)
            return False
        return self._release(turn, result)
