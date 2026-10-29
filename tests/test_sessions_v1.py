import threading
from alter_ego.sessions import SessionDialogs
from test_dialog import make_orch,FakeRegistry

def test_history_isolation_a_b_a_and_same_name(tmp_path):
 regs=[]
 def factory():
  reg=FakeRegistry(['{"ja":"はい。","zh":"是的。","emotion":"neutral"}']);regs.append(reg)
  return make_orch(tmp_path,reg)
 m=SessionDialogs(factory)
 a=m.arrive('A','同名');m.reply(a,'A秘密一')
 b=m.arrive('B','同名');m.reply(b,'B秘密一')
 a=m.arrive('A','改显示名');m.reply(a,'A二')
 assert 'A秘密一' in str(regs[0].calls[-1]);assert 'B秘密一' not in str(regs[0].calls[-1])
 assert 'A秘密一' not in str(regs[1].calls[-1])
 m.reset(a.key);m.reply(m.snapshot(),'A新');assert len(regs)==3
 assert regs[1] is b.dialog.registry

def test_guests_expiry_eviction_and_late_result(tmp_path):
 clock=[0.]; regs=[]
 def factory():
  reg=FakeRegistry(['{"ja":"はい。","zh":"是的。","emotion":"neutral"}']);regs.append(reg)
  return make_orch(tmp_path,reg)
 m=SessionDialogs(factory,max_sessions=2,idle_ttl=10,clock=lambda:clock[0])
 g=m.arrive(name='访客');m.reply(g,'guest secret');m.leave(); g2=m.arrive(name='访客')
 assert g2.key!=g.key;assert g.key not in m.entries
 m.reply(g2,'new guest');u=m.arrive('A','访客');m.reply(u,'A');assert 'guest secret' not in str(regs[-1].calls)
 m.arrive('B');assert len(m.entries)==2
 clock[0]=20;m.arrive('C');assert len(m.entries)==1
 started=threading.Event();done=threading.Event()
 old=m.snapshot();original=old.dialog.registry.chat_with_fallback
 def slow(*a,**kw):started.set();done.wait(5);return original(*a,**kw)
 old.dialog.registry.chat_with_fallback=slow;result=[]
 t=threading.Thread(target=lambda:result.append(m.reply(old,'迟到内容')));t.start();assert started.wait(5)
 m.arrive('D');done.set();t.join(5);assert not t.is_alive();assert result==[None]

def test_history_six_rounds_and_reset_invalidates_snapshot(tmp_path):
 m=SessionDialogs(lambda:make_orch(tmp_path,FakeRegistry(['{"ja":"はい。","zh":"是。","emotion":"neutral"}'])))
 s=m.arrive('A')
 for i in range(10):m.reply(s,str(i))
 assert s.dialog.history_turns==6
 m.reset(s.key);assert m.reply(s,'stale') is None
