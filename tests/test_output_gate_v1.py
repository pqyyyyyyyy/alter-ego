import json
import pytest
from alter_ego.dialog import parse_cloud_reply
from alter_ego.persona.output_gate import japanese_error,SAFE_JA
from test_dialog import make_orch,FakeRegistry,CARD_JA
@pytest.mark.parametrize('raw',['[]','42','null','true','{"ja":[]}','{"ja":23}',
 '{"ja":"你好。"}','{"ja":"hello"}','{"ja":""}','{"ja":"はい。","zh":{}}',
 '{"ja":"'+('あ'*2001)+'"}','{"ja":"\\ud800"}'])
def test_invalid_never_reaches_tts(raw,tmp_path):
 reply,note=parse_cloud_reply(raw);assert not reply.ja;assert note
 turn=make_orch(tmp_path,FakeRegistry([raw])).reply('虚构输入')
 assert not japanese_error(turn.speak_ja);assert turn.error
@pytest.mark.parametrize('text',['了解。','今日は良い天気です。','大丈夫。','AIを使います。'])
def test_kanji_japanese_allowed(text):assert not japanese_error(text)
def test_missing_subtitle_and_partial_language_limit():
 r,n=parse_cloud_reply('{"ja":"はい。","emotion":"weird"}')
 assert r.zh=='';assert '字幕缺失' in n;assert r.emotion=='neutral'
 # Heuristic limitation documented: mixed Chinese with a kana can pass, NOT semantic proof.
 assert not japanese_error('你好ね')
def test_bad_preset_safe_and_late_budget_discard(tmp_path,monkeypatch):
 card={**CARD_JA,'profile':{**CARD_JA['profile'],'fallback_reply':'中文'}}
 turn=make_orch(tmp_path,FakeRegistry(['{}']),card).reply('虚构')
 assert turn.speak_ja==SAFE_JA;assert '预设' in turn.error
 orch=make_orch(tmp_path,FakeRegistry(['{"ja":"はい。"}'])); orch.cloud_budget_s=1
 clock=iter([0,2]);monkeypatch.setattr('alter_ego.dialog.time.perf_counter',lambda:next(clock))
 turn=orch.reply('虚构');assert '预算超限' in turn.error


@pytest.mark.parametrize('error',['401 invalid API key','400 Bad Request invalid user input','422 invalid model','unsupported transport'])
def test_unrelated_parameter_or_auth_failure_not_retried(tmp_path,error):
 reg=FakeRegistry(raise_with=RuntimeError(error));turn=make_orch(tmp_path,reg).reply('虚构输入')
 assert len(reg.calls)==1;assert turn.error
