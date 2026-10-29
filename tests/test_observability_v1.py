import json
import pytest
from alter_ego.observability import EventLog,summarize,statistics_report,compare

def fixture(device='pc',clock='pc-process'):
    l=EventLog(device,clock,clock=lambda:0)
    for event,mono,data in [('turn_start',0,{}),('vad_end',1,{}),('asr_start',1,{'input_audio_s':2}),('asr_end',2,{}),('cloud_start',2,{}),('cloud_end',3,{}),('rewrite_start',3,{}),('rewrite_end',3.1,{}),('tts_start',3.1,{}),('tts_end',4.1,{'output_audio_s':4}),('ws_sent',4.2,{}),('audio_started',4.5,{}),('audio_ended',8.5,{'result':'ended'})]:
        r=l.add(event,'t','s',**data);r['mono_s']=mono
    return l

def test_rtf_jsonl_and_no_private_fields(tmp_path):
    l=fixture();s=summarize(l.rows)[0]
    assert s['asr_rtf']==.5;assert s['tts_rtf']==pytest.approx(.25)
    assert s['stop_to_first_audio_sec']==3.5;assert s['turn_sec']==8.5
    assert s['complete'];l.write_jsonl(tmp_path/'events.jsonl')
    assert len((tmp_path/'events.jsonl').read_text(encoding='utf8').splitlines())==13
    with pytest.raises(ValueError):l.add('asr_end','t',raw_user_text='secret')

def test_clocks_devices_missing_failures_and_order():
    a=fixture();b=fixture('board','board-process');c=fixture('pc','browser-process')
    assert len(summarize(a.rows+b.rows+c.rows))==3
    assert summarize(list(reversed(a.rows)))[0]['turn_sec']==8.5
    a.rows=[r for r in a.rows if r['event']!='tts_end'];assert summarize(a.rows)[0]['tts_rtf'] is None
    a.add('failed','t','s',reason='fixture');assert summarize(a.rows)[0]['failed']
    b.rows[3]['mono_s']=.5;assert summarize(b.rows)[0]['asr_sec'] is None
    c.rows.append(c.rows[3]);assert not summarize(c.rows)[0]['complete']

def test_stats_include_failures_and_workload_equality():
    good=summarize(fixture().rows)[0];bad={**good,'failed':True,'complete':False,'tts_sec':None}
    report=statistics_report([good,bad]);assert report['n_total']==2 and report['n_failed']==1
    assert report['metrics']['tts_sec']['n_missing']==1
    assert report['metrics']['tts_sec']['p95'] is None
    meta=dict(device_id='pc',version='v1',backend='fake',cold=False,input_manifest_sha256='fake',seed=42,n_planned=2)
    assert compare([good,bad],[good,bad],meta,meta)
    with pytest.raises(ValueError):compare([good,bad],[good],meta,meta)
    with pytest.raises(ValueError):compare([good,bad],[good,bad],meta,{**meta,'device_id':'board'})


def test_statistics_refuse_cross_device_or_browser_clock_mix():
    a=summarize(fixture().rows)[0];b=summarize(fixture('board','board-clock').rows)[0]
    with pytest.raises(ValueError):statistics_report([a,b])
    c=summarize(fixture('pc','browser-clock').rows)[0]
    with pytest.raises(ValueError):statistics_report([a,c])


def test_playback_trace_and_failed_or_degraded_work_are_separate():
    log=fixture()
    log.add('failed','t','s',reason='cloud_error')
    log.add('resources','t','s',result='degraded')
    sample=summarize(log.rows)[0]
    assert sample['trace_complete'] and sample['playback_complete']
    assert sample['failed'] and sample['degraded'] and not sample['complete']
    assert statistics_report([sample])['n_degraded']==1
