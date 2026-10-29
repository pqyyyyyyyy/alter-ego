"""Voice warmup and modes via real local HTTP/WS; no hardware or API keys."""
import importlib
import json
import urllib.error

import pytest

from test_server_integration import client, until


def switch(client, enabled, mode='fast'):
    return client.request('/voice/control', data={'enabled': enabled, 'mode': mode},
                          headers={'X-Alter-Ego-Voice': '1', 'Origin': client.url})


@pytest.mark.parametrize('mode', ['fast', 'sync'])
def test_text_delivery_waits_for_audio_only_in_sync(client, mode):
    with client.websocket() as (ws, hello):
        assert switch(client, True, mode)['ok']
        client.control(block='tts')
        pending = client.pool.submit(client.chat)
        until(lambda: client.request('/debug/fixture/control')['started']['tts'])
        if mode == 'fast':
            preview = client.receive(ws, 'reply_preview')
            assert preview['text'] and 'audio' not in preview
        else:
            while True:
                try:
                    message = json.loads(ws.recv(timeout=.03))
                except TimeoutError:
                    break
                assert message['type'] not in ('speak', 'reply_preview')
        client.control(release='tts')
        assert pending.result(timeout=3)['ok']
        speech = client.receive(ws)
        assert speech['audio'].startswith('data:audio/wav;base64,')
        if mode == 'fast':
            assert speech['speech_id'] == preview['speech_id']
        client.ack(ws, speech)
        client.idle()
        events = client.request('/debug/events')['rows']
        assert sum(row['event'] == 'text_sent' for row in events) == (mode == 'fast')


@pytest.mark.parametrize('failure', [False, True])
def test_real_warmup_gates_input_then_releases_and_only_runs_once(client, failure):
    module = importlib.import_module('alter_ego.server.app')
    voice = module.rt.lifecycle.voice_output
    voice.enabled = voice.warmed = False
    client.control(block='tts', tts_error=failure)
    with client.websocket() as (ws, hello):
        assert switch(client, True)['ok']
        until(lambda: client.request('/debug/fixture/control')['started']['tts'])
        assert client.health()['voice_output']['status'] == 'warming'
        assert not client.health()['policy']['voice_submit']
        assert client.chat()['error'] == 'tts_warming'
        assert client.request('/debug/transcribe', data=b'\0' * 640)['error'] == 'tts_warming'
        assert not switch(client, True)['ok']
        # Muting does not terminate the native warmup or reopen the input gate.
        assert switch(client, False)['ok']
        assert client.health()['voice_output']['status'] == 'warming'
        client.control(release='tts')
        until(lambda: client.health()['voice_output']['status'] != 'warming')
        assert client.health()['policy']['voice_submit']
        assert client.request('/debug/fixture/control')['calls']['cloud'] == 0
        assert client.request('/debug/fixture/control')['calls']['tts'] == 1
        if failure:
            assert client.health()['voice_output']['status'] == 'failed'
            assert not client.health()['voice_output']['enabled']
            assert client.chat()['ok']
            speech = client.receive(ws)
            assert speech['audio'] is None
            client.ack(ws, speech, 'no_audio')
        else:
            assert switch(client, True)['ok']
            assert client.health()['voice_output']['status'] == 'ready'
            assert switch(client, False)['ok']
            assert switch(client, True, 'sync')['ok']
            assert client.request('/debug/fixture/control')['calls']['tts'] == 1


def test_off_mutes_without_releasing_unfinished_synthesis(client):
    with client.websocket() as (ws, hello):
        client.control(block='tts')
        pending = client.pool.submit(client.chat)
        until(lambda: client.request('/debug/fixture/control')['started']['tts'])
        assert switch(client, False)['ok']
        assert client.health()['active'] is not None
        assert client.chat()['status'] == 'busy'
        client.control(release='tts')
        assert pending.result(timeout=3)['ok']
        speech = client.receive(ws)
        assert speech['audio'] is None
        client.ack(ws, speech, 'no_audio')
        client.idle()
        assert client.chat()['ok']
        speech = client.receive(ws)
        assert speech['audio'] is None
        assert client.request('/debug/fixture/control')['calls']['tts'] == 1
        client.ack(ws, speech, 'no_audio')


@pytest.mark.parametrize('data,headers,code', [
    ({'enabled': True, 'mode': 'fast'}, {}, 403),
    ({'enabled': True, 'mode': 'fast'}, {'X-Alter-Ego-Voice': '1', 'Origin': 'http://other'}, 403),
    ({'enabled': 'yes', 'mode': 'fast'}, {'X-Alter-Ego-Voice': '1'}, 400),
    ({'enabled': True, 'mode': 'other'}, {'X-Alter-Ego-Voice': '1'}, 400),
])
def test_voice_control_rejects_invalid_or_cross_origin(client, data, headers, code):
    with pytest.raises(urllib.error.HTTPError) as error:
        client.request('/voice/control', data=data, headers=headers)
    assert error.value.code == code


def test_real_app_voice_defaults_off_and_cannot_enable_without_tts():
    from alter_ego.server.voice_output import VoiceOutput
    from types import SimpleNamespace
    import asyncio
    runtime = SimpleNamespace(tts=None, fixture=False)
    lifecycle = SimpleNamespace(rt=runtime, closed=False, coordinator=SimpleNamespace(active=None))
    manager = VoiceOutput(lifecycle, None)
    assert manager.snapshot()['status'] == 'off'
    assert asyncio.run(manager.control(True, 'fast'))['error'] == 'tts_not_configured'
