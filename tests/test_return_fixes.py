import struct

import pytest

from alter_ego.observability import EventLog, summarize, statistics_report
from alter_ego.persona.output_gate import SAFE_JA
from alter_ego.sessions import SessionDialogs
from alter_ego.voice.input_controller import PCMConfig, Segmenter
from test_dialog import FakeRegistry, make_orch


@pytest.mark.parametrize('raw', [
    '{"zh":"unrelated","emotion":"happy"}',
    '{"ja":"","zh":"unrelated"}',
    '{"ja":123,"zh":"unrelated"}',
    '{"ja":null,"zh":"unrelated"}',
])
@pytest.mark.parametrize('bad_preset', [False, True])
def test_fallback_never_pairs_unrelated_subtitle(tmp_path, raw, bad_preset):
    dialog = make_orch(tmp_path, FakeRegistry([raw]))
    if bad_preset:
        dialog.card.profile.fallback_reply = 'invalid preset'
    turn = dialog.reply('test')
    assert turn.subtitle_zh == ''
    assert turn.emotion == 'neutral'
    assert turn.error and dialog.history_turns == 0
    if bad_preset:
        assert turn.speak_ja == SAFE_JA


@pytest.mark.parametrize('action', ['switch', 'roundtrip', 'reset', 'ttl', 'leave', 'guest_leave'])
def test_stale_snapshot_never_calls_model(tmp_path, action):
    clock = [0.]
    manager = SessionDialogs(lambda: make_orch(tmp_path, FakeRegistry(['{"ja":"はい。"}'])),
                             idle_ttl=10, clock=lambda: clock[0])
    snapshot = manager.arrive(None if action == 'guest_leave' else 'A')
    if action in ('switch', 'roundtrip'):
        manager.arrive('B')
        if action == 'roundtrip': manager.arrive('A')
    elif action == 'reset': manager.reset(snapshot.key)
    elif action == 'ttl': clock[0] = 11
    else: manager.leave()
    assert manager.reply(snapshot, 'stale') is None
    assert snapshot.dialog.registry.calls == []
    assert snapshot.dialog.history_turns == 0


@pytest.mark.parametrize('stage', ['asr', 'cloud', 'rewrite', 'tts'])
@pytest.mark.parametrize('endpoint', ['start', 'end'])
def test_missing_stage_endpoint_is_incomplete(stage, endpoint):
    log = EventLog('pc', 'fake', clock=lambda: 0)
    log.add('turn_start', 't')
    log.add(stage + '_' + endpoint, 't')
    log.add('audio_ended', 't', result='ended')
    sample = summarize(log.rows)[0]
    missing = 'end' if endpoint == 'start' else 'start'
    assert 'missing:' + stage + '_' + missing in sample['errors']
    assert sample[stage + '_sec'] is None
    assert sample['playback_complete'] and not sample['complete']
    assert statistics_report([sample])['n_incomplete'] == 1


def test_unused_stages_may_be_absent():
    log = EventLog('pc', 'fake', clock=lambda: 0)
    for event in ('turn_start', 'tts_start', 'tts_end', 'audio_ended'):
        log.add(event, 't')
    assert summarize(log.rows)[0]['complete']


@pytest.mark.parametrize('pre_ms', [0, 20, 40, 60, 100, 200])
@pytest.mark.parametrize('chunk', [1, 3, 319, 640, 997])
def test_confirmation_frames_preserve_sentence_onset(pre_ms, chunk):
    frames = [struct.pack('<320h', *([v] * 320)) for v in (1000, 2000, 3000)]
    pcm = b''.join(frames) + bytes(1280)
    segmenter = Segmenter(PCMConfig(pre_ms=pre_ms, start_ms=60, silence_ms=40,
                                    min_ms=60, max_ms=1000))
    results = []
    for i in range(0, len(pcm), chunk):
        results.extend(segmenter.feed(pcm[i:i + chunk]))
    assert results == [pcm]
