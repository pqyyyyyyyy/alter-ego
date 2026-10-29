import asyncio
import pytest
from alter_ego.scheduler.coordinator import Coordinator

@pytest.mark.parametrize("a,b",[("chat","chat"),("chat","say"),("voice","chat"),("say","voice")])
def test_single_reservation(a,b):
 c=Coordinator(); t=c.reserve(a,"A"); assert t; assert c.reserve(b,"B") is None
 assert not c.budget().voice_submit
 assert not c.budget().identity_extract
 assert c.queue_limit <= 1

def test_playback_ids_and_recovery():
 async def run():
  c=Coordinator(); t=c.reserve("chat","A")
  async def job(): return "fake"
  assert (await c.run(t,job))["status"]=="playback"
  assert c.state=="SPEAKING"
  assert not c.acknowledge("B",t.turn_id,t.speech_id)
  assert not c.acknowledge("A","old",t.speech_id)
  assert c.acknowledge("A",t.turn_id,t.speech_id)
  assert not c.acknowledge("A",t.turn_id,t.speech_id)
  n=c.reserve("say","A"); assert not c.acknowledge("A",t.turn_id,t.speech_id)
  assert c.active is n
 asyncio.run(run())

@pytest.mark.parametrize("reason",["timeout","disconnect","left","caller_cancelled"])
def test_late_worker_holds_gate(reason):
 async def run():
  clock=[0.]; c=Coordinator(clock=lambda:clock[0],model_timeout=1)
  started=asyncio.Event(); done=asyncio.Event(); running=[0]; peak=[0]
  async def job():
   running[0]+=1; peak[0]=max(peak[0],running[0]); started.set()
   await done.wait(); running[0]-=1; return "late"
  t=c.reserve("chat","A"); pending=asyncio.create_task(c.run(t,job)); await started.wait()
  if reason=="timeout": clock[0]=2; assert c.expire()
  elif reason=="disconnect": assert c.disconnect("A")
  elif reason=="left": c.leave()
  else:
   pending.cancel()
   with pytest.raises(asyncio.CancelledError): await pending
  assert c.reserve("say","A") is None; assert c.active; assert not c.budget().voice_submit
  done.set()
  if reason!="caller_cancelled": assert (await pending)["status"]=="discarded"
  else: await t.worker; await asyncio.sleep(0)
  assert c.active is None; assert peak[0]==1; assert c.reserve("chat")
 asyncio.run(run())

def test_errors_no_client_playback_timeout_baseline():
 async def run():
  c=Coordinator()
  async def bad(): raise RuntimeError("fake cloud failure")
  assert (await c.run(c.reserve("chat"),bad))["status"]=="failed"
  async def ok():return "ok"
  assert (await c.run(c.reserve("say"),ok))["status"]=="no_client"
  clock=[0]; c=Coordinator(clock=lambda:clock[0],playback_timeout=1)
  await c.run(c.reserve("chat","A"),ok); clock[0]=2; c.expire(); assert c.active is None
  c.reserve("chat","A"); assert not c.budget().identity_extract
  c.baseline=True; assert c.budget().identity_extract; assert not c.budget().voice_submit
  assert c.budget().presence_fps==5
 asyncio.run(run())


def test_synchronous_thread_inference_not_released_by_caller_cancel():
 import threading
 async def run():
  c=Coordinator();started=threading.Event();finished=threading.Event()
  def sync_inference():
   started.set();assert finished.wait(5);return 'late-thread'
  async def job():return await asyncio.to_thread(sync_inference)
  t=c.reserve('chat','A');pending=asyncio.create_task(c.run(t,job))
  assert await asyncio.to_thread(started.wait,5);pending.cancel()
  with pytest.raises(asyncio.CancelledError):await pending
  assert c.active is t and t.phase=='quarantine';assert c.reserve('say','A') is None
  finished.set();await t.worker;await asyncio.sleep(0)
  assert c.active is None;assert c.reserve('chat','A')
 asyncio.run(run())


@pytest.mark.parametrize('action',['failed','disconnect','left'])
def test_speaking_failure_release_and_next_request(action):
 async def run():
  c=Coordinator();t=c.reserve('chat','A')
  async def job():return 'fake synthesis'
  await c.run(t,job)
  if action=='failed':assert c.acknowledge('A',t.turn_id,t.speech_id,'failed')
  elif action=='disconnect':assert c.disconnect('A')
  else:assert c.leave()
  assert c.active is None;assert c.budget().voice_submit;assert c.budget().identity_extract
  assert c.reserve('say','A')
 asyncio.run(run())
