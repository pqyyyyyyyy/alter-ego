"""Model-free metrics. No raw audio, full personal text or secrets in records."""
from __future__ import annotations
import csv
import json
import math
import statistics
import time
import threading
from collections import defaultdict
from pathlib import Path

EVENTS = frozenset(('turn_start','vad_end','asr_start','asr_end','cloud_start','cloud_end','rewrite_start','rewrite_end','tts_start','tts_end','text_sent','ws_sent','audio_started','audio_ended','failed','rejected','timeout','state','gate','resources'))
SAFE_FIELDS = frozenset(('input_audio_s','output_audio_s','result','reason','state','enabled','cpu_percent','rss_bytes','threads','queue_depth','npu_watts','client_elapsed_ms','backend','cold','seed'))
class EventLog:
    def __init__(self, device_id, clock_id, clock=time.monotonic, max_rows=50000):
        self.device_id,self.clock_id,self.clock=device_id,clock_id,clock
        self.rows=[]
        self.lock = threading.Lock()
        self.max_rows = max_rows
    def add(self,event,turn_id,speech_id=None,**data):
        if event not in EVENTS: raise ValueError('unknown event')
        if set(data)-SAFE_FIELDS: raise ValueError('unapproved/private field')
        row=dict(schema='aeg-events-v1',device_id=self.device_id,clock_id=self.clock_id,turn_id=turn_id,speech_id=speech_id,event=event,mono_s=self.clock(),wall_time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),data=data)
        with self.lock:
            self.rows.append(row)
            if len(self.rows) > self.max_rows:
                del self.rows[:len(self.rows)-self.max_rows]
        return row

    def snapshot(self):
        with self.lock:
            return list(self.rows)
    def write_jsonl(self,path):
        Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in self.rows),encoding='utf-8')

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        if r.get('schema')!='aeg-events-v1': raise ValueError('schema mismatch')
        groups[(r['device_id'],r['clock_id'],r['turn_id'])].append(r)
    output=[]
    for (device,clock,turn),rs in groups.items():
        errors=[]; by=defaultdict(list)
        for r in rs: by[r['event']].append(r)
        def one(event):
            vals=by[event]
            if len(vals)>1: errors.append('duplicate:'+event); return None
            return vals[0] if vals else None
        for start,end in (('asr_start','asr_end'),('cloud_start','cloud_end'),
                          ('rewrite_start','rewrite_end'),('tts_start','tts_end')):
            if bool(by[start]) != bool(by[end]):
                errors.append('missing:'+(end if by[start] else start))
        def delta(a,b):
            x,y=one(a),one(b)
            if x is None or y is None:return None
            # Only one clock/device per group. Browser clocks never join board groups.
            value=y['mono_s']-x['mono_s']
            if not math.isfinite(value) or value<0:errors.append('invalid_order:'+a+':'+b);return None
            return value
        def duration(key):
            values=[r.get('data',{}).get(key) for r in rs if key in r.get('data',{})]
            if not values:return None
            if len(set(values))!=1:errors.append('conflicting:'+key);return None
            x=values[0]
            return x if isinstance(x,(int,float)) and math.isfinite(x) and x>0 else None
        asr=delta('asr_start','asr_end');tts=delta('tts_start','tts_end')
        ai,ao=duration('input_audio_s'),duration('output_audio_s')
        for required in ('turn_start','audio_ended'):
            if not by[required]: errors.append('missing:'+required)
        speechs={r.get('speech_id') for r in rs if r.get('speech_id')}
        if len(speechs)>1:errors.append('multiple_speech_ids')
        ended=one('audio_ended')
        degraded=any(r.get('data',{}).get('result')=='degraded' for r in rs)
        failed=bool(by['failed'] or by['timeout'] or by['rejected'] or (ended and ended.get('data',{}).get('result','ended')!='ended'))
        output.append(dict(device_id=device,clock_id=clock,turn_id=turn,speech_id=next(iter(speechs)) if len(speechs)==1 else None,
            asr_sec=asr,cloud_sec=delta('cloud_start','cloud_end'),rewrite_sec=delta('rewrite_start','rewrite_end'),tts_sec=tts,
            asr_rtf=asr/ai if asr is not None and ai else None,tts_rtf=tts/ao if tts is not None and ao else None,
            stop_to_first_audio_sec=delta('vad_end','audio_started'),turn_sec=delta('turn_start','audio_ended'),
            failed=failed,degraded=degraded,trace_complete=not errors,
            playback_complete=bool(ended and ended.get('data',{}).get('result','ended')=='ended'),
            complete=ended is not None and not failed and not errors,errors=errors))
    return output

def statistics_report(samples, fields=('asr_sec','tts_sec','asr_rtf','tts_rtf','turn_sec'), min_p95=20):
    if len({(s['device_id'],s['clock_id']) for s in samples}) > 1:
        raise ValueError('statistics must not mix devices/clocks')
    result=dict(n_total=len(samples),n_failed=sum(s['failed'] for s in samples),
                n_degraded=sum(s.get('degraded',False) for s in samples),
                n_incomplete=sum(not s['complete'] for s in samples),metrics={})
    for field in fields:
        values=sorted(s[field] for s in samples if s.get(field) is not None)
        result['metrics'][field]=dict(n_observed=len(values),n_missing=len(samples)-len(values),median=statistics.median(values) if values else None,min=min(values) if values else None,max=max(values) if values else None,p95=values[math.ceil(.95*len(values))-1] if len(values)>=min_p95 else None)
    return result

def compare(left,right,left_meta,right_meta):
    keys=('device_id','version','backend','cold','input_manifest_sha256','seed','n_planned')
    if any(left_meta.get(k)!=right_meta.get(k) for k in keys):raise ValueError('unequal workload/environment')
    if any(s['device_id'] != left_meta['device_id'] for s in left+right):raise ValueError('sample/meta device mismatch')
    if len(left)!=left_meta['n_planned'] or len(right)!=right_meta['n_planned']:raise ValueError('missing planned samples')
    return {'left':statistics_report(left),'right':statistics_report(right),'metadata':left_meta}

def write_csv(samples,path):
    if not samples:raise ValueError('no samples')
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
