"""In-memory bounded stable-ID sessions. No persistence or personal-text logging.
Each entry owns a separate DialogOrchestrator; P13 must serialize model work.
"""
from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass
from collections import OrderedDict

@dataclass(frozen=True)
class Snapshot:
    key: str
    epoch: int
    generation: int
    dialog: object
    user_name: str | None

class SessionDialogs:
    def __init__(self, factory, max_sessions=16, idle_ttl=900., clock=time.monotonic):
        if max_sessions<1 or idle_ttl<=0:raise ValueError('invalid limits')
        self.factory,self.max_sessions,self.ttl,self.clock=factory,max_sessions,idle_ttl,clock
        self.entries=OrderedDict(); self.current=None;self.epoch=0;self.generation=0
        self.lock=threading.RLock(); self.inflight=set()
    def _prune(self):
        now=self.clock()
        for k,(d,when,g) in list(self.entries.items()):
            if now-when>=self.ttl and k not in self.inflight:
                del self.entries[k]
    def arrive(self, user_id=None, name=None):
        with self.lock:
            self._prune(); key='user:'+str(user_id) if user_id else 'guest:'+uuid.uuid4().hex
            if key not in self.entries:
                while len(self.entries)>=self.max_sessions:
                    victim=next((k for k in self.entries if k not in self.inflight),None)
                    if victim is None:raise RuntimeError('all sessions busy')
                    del self.entries[victim]
                self.generation+=1
                self.entries[key]=(self.factory(),self.clock(),self.generation)
            if self.current!=key:self.epoch+=1
            self.current=key
            d,_,g=self.entries[key];self.entries[key]=(d,self.clock(),g); self.entries.move_to_end(key)
            return Snapshot(key,self.epoch,g,d,name)
    def snapshot(self, name=None):
        with self.lock:
            self._prune()
            if self.current not in self.entries:raise RuntimeError('no active session')
            d,_,g=self.entries[self.current]
            return Snapshot(self.current,self.epoch,g,d,name)
    def leave(self):
        with self.lock:
            if self.current and self.current.startswith('guest:'):self.entries.pop(self.current,None)
            self.current=None;self.epoch+=1
    def reset(self,key):
        with self.lock:
            if key in self.entries:
                self.generation+=1
                self.entries[key]=(self.factory(),self.clock(),self.generation)
                if key==self.current:self.epoch+=1
    def is_current(self, snapshot):
        with self.lock:
            self._prune()
            entry = self.entries.get(snapshot.key)
            return bool(entry and entry[2] == snapshot.generation and
                        snapshot.epoch == self.epoch and snapshot.key == self.current)

    def reply(self,snapshot,text,ctx=None,event_hook=None):
        with self.lock:
            self._prune()
            entry=self.entries.get(snapshot.key)
            if (snapshot.epoch!=self.epoch or snapshot.key!=self.current or
                not entry or entry[2]!=snapshot.generation):return None
            if snapshot.key in self.inflight:raise RuntimeError('session busy')
            self.inflight.add(snapshot.key)
        try:
            if ctx is not None:
                from dataclasses import replace
                ctx=replace(ctx,user_name=snapshot.user_name)
            if event_hook is None:
                result=snapshot.dialog.reply(text,ctx)
            else:
                result=snapshot.dialog.reply(text,ctx,event_hook=event_hook)
        finally:
            with self.lock:self.inflight.discard(snapshot.key)
        with self.lock:
            entry=self.entries.get(snapshot.key)
            if entry and entry[2]==snapshot.generation:
                self.entries[snapshot.key]=(entry[0],self.clock(),entry[2])
            if (snapshot.epoch!=self.epoch or snapshot.key!=self.current or
                not entry or entry[2]!=snapshot.generation):return None
            return result
