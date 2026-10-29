import asyncio
import struct
import threading
import pytest
from alter_ego.voice.input_controller import PCMConfig,Segmenter,InputController
F=struct.pack('<320h',*([1000]*320)); Z=bytes(640)
def cfg():return PCMConfig(pre_ms=100,start_ms=40,silence_ms=40,min_ms=60,max_ms=1000)
def test_arbitrary_chunks_same_sentence_prefix():
 pcm=Z*3+F*8+Z*3
 expected=Segmenter(cfg()).feed(pcm)
 for size in [1,3,319,640,997,100000]:
  s=Segmenter(cfg()); actual=[]
  for i in range(0,len(pcm),size):actual+=s.feed(pcm[i:i+size])
  assert actual==expected; assert F in actual[0]; assert len(actual)==1
 assert expected[0].startswith(Z*3+F*2)
def test_silence_short_noise_two_sentences():
 s=Segmenter(cfg()); assert s.feed(Z*100)==[]; assert s.feed(F+Z*10)==[]
 assert len(s.feed((F*8+Z*4)*2))==2
 assert s.feed(F*2+Z*4)==[]
def test_limit_gate_eof_and_validation():
 s=Segmenter(PCMConfig()); parts=s.feed(F*1600); assert all(len(p)<=960000 for p in parts); assert parts
 s.gate(True); assert s.feed(F*100)==[]; s.gate(False); assert s.feed(Z*30)==[]
 s.feed(b'x')
 with pytest.raises(ValueError):s.end_stream()
 with pytest.raises(TypeError):s.feed('bad')
 with pytest.raises(ValueError):PCMConfig(max_ms=31000)
def test_gate_mid_sentence_starts_clean():
 s=Segmenter(cfg()); s.feed(F*5); s.gate(True); s.feed(F*8); s.gate(False)
 assert not s.feed(Z*5); assert len(s.feed(F*8+Z*5))==1

class ASR:
 def __init__(self):self.calls=0; self.reset_calls=0
 def reset(self):self.reset_calls+=1
 def feed(self,pcm):self.pcm=pcm
 def poll(self):
  self.calls+=1
  if self.calls==1:raise RuntimeError('fake failure')
  return '虚构识别结果'
def test_asr_error_and_next_sentence_once():
 async def run():
  a=ASR(); got=[]; c=InputController(a,lambda t,i:got.append((t,i)),Segmenter(cfg()))
  assert await c.start(); assert not await c.start()
  for _ in range(2):c.feed(F*8+Z*5); await c.queue.join()
  assert a.calls==2; assert len(got)==1; assert a.reset_calls==4
  await c.stop(); await c.stop(); assert await c.start(); await c.stop()
 asyncio.run(run())
def test_inflight_gate_and_stop_no_stale_submission():
 async def run():
  started=threading.Event(); done=threading.Event(); got=[]
  class Slow(ASR):
   def poll(self):started.set(); done.wait(5);return '旧语音'
  c=InputController(Slow(),lambda t,i:got.append(t),Segmenter(cfg())); await c.start()
  c.feed(F*8+Z*5); await asyncio.to_thread(started.wait,5)
  c.gate(True);c.gate(False);done.set();await c.queue.join();assert not got
  c.feed(F*8+Z*5);await c.queue.join();assert got==['旧语音']
  await c.stop()
 asyncio.run(run())


def test_stop_during_blocking_asr_waits_and_suppresses_text():
 async def run():
  started=threading.Event();done=threading.Event();got=[]
  class Slow(ASR):
   def poll(self):started.set();assert done.wait(5);return 'stale'
  c=InputController(Slow(),lambda t,i:got.append(t),Segmenter(cfg()));await c.start()
  c.feed(F*8+Z*5);assert await asyncio.to_thread(started.wait,5)
  stopping=asyncio.create_task(c.stop());await asyncio.sleep(0)
  assert not c.running;assert not await c.start();assert not stopping.done()
  done.set();await stopping;assert got==[];await c.stop()
 asyncio.run(run())


@pytest.mark.parametrize('value,expected', [(('final',True),'final'),(('partial',False),''),
                                          (('',True),''),(None,''),('  text  ','text')])
def test_real_asr_poll_contract(value,expected):
 class Adapter(ASR):
  def poll(self):return value
 c=InputController(Adapter(),lambda *args:None)
 assert c._transcribe(F)==expected
