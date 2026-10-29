"""Real loopback HTTP/WS tests; no TestClient/httpx, device, model or paid API."""
import base64
import concurrent.futures
import importlib
import json
import socket
import struct
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager

import pytest
import uvicorn
from websockets.sync.client import connect

from alter_ego.observability import summarize


def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result: return result
        time.sleep(.01)
    raise AssertionError('condition did not become true')


class Client:
    def __init__(self, port, server):
        self.url = f'http://127.0.0.1:{port}'
        self.ws_url = f'ws://127.0.0.1:{port}/ws'
        self.server = server
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def request(self, path, *, data=None, params=None, post=False, headers=None):
        if params: path += '?' + urllib.parse.urlencode(params)
        body = data if isinstance(data, bytes) else json.dumps(data).encode() if data is not None else b''
        req = urllib.request.Request(self.url + path, data=body if post or data is not None else None,
                                     headers={**({'Content-Type': 'application/json'} if data is not None else {}), **(headers or {})})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read())

    def chat(self, text='fixture request', path='/debug/chat'):
        return self.request(path, params={'text': text}, post=True)

    def control(self, **data): return self.request('/debug/fixture/control', data=data)
    def health(self): return self.request('/health')
    def idle(self): return until(lambda: self.health()['active'] is None)

    @contextmanager
    def websocket(self):
        with connect(self.ws_url, proxy=None, open_timeout=3) as ws:
            hello = json.loads(ws.recv(timeout=3))
            assert hello['type'] == 'hello' and hello['protocol'] == 'aeg-ws-v2'
            assert json.loads(ws.recv(timeout=3))['type'] == 'sync'
            yield ws, hello

    def receive(self, ws, kind='speak'):
        for _ in range(40):
            msg = json.loads(ws.recv(timeout=3))
            if msg['type'] == kind: return msg
        raise AssertionError('message not received: ' + kind)

    def ack(self, ws, speech, result='ended', **kwargs):
        ws.send(json.dumps({'type': 'audio_ended', 'turn_id': speech['turn_id'],
                            'speech_id': speech['speech_id'], 'result': result, **kwargs}))


@pytest.fixture
def client(monkeypatch):
    module = importlib.import_module('alter_ego.server.app')
    monkeypatch.setattr(module, 'rt', module.Runtime())
    monkeypatch.setenv('ALTER_EGO_FIXTURE', '1')
    monkeypatch.setenv('ALTER_EGO_MODEL_TIMEOUT', '.35')
    monkeypatch.setenv('ALTER_EGO_DEVICE_ID', 'pc-integration-test')
    monkeypatch.setenv('ALTER_EGO_STOP_TIMEOUT', '.15')
    monkeypatch.setattr(module, 'ProviderRegistry', lambda *args: pytest.fail('fixture accessed real provider'))
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(module.app, host='127.0.0.1', port=port,
                                         log_level='error', ws='websockets-sansio'))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]))
    thread.start()
    c = Client(port, server)
    until(lambda: server.started)
    try:
        assert module.rt.camera is None
        yield c
    finally:
        for event in module.rt.fixture_control.releases.values(): event.set()
        server.should_exit = True
        thread.join(5)
        c.pool.shutdown(wait=True)
        assert not thread.is_alive(), 'service worker/device shutdown did not complete'
        sock.close()
        probe = socket.socket()
        try: assert probe.connect_ex(('127.0.0.1', port)) != 0
        finally: probe.close()


def test_camera_metadata_synced_on_websocket_join(client, monkeypatch):
    module = importlib.import_module('alter_ego.server.app')
    monkeypatch.setattr(module.rt, 'camera_view', {
        'schema': 'h04-camera-v1', 'session_id': 'test', 'available': True,
        'status': 'ready', 'face_count': 2, 'visible_users': ['A', 'B'], 'frame_age_s': .1})
    with client.websocket() as (ws, hello):
        presence = client.receive(ws, 'presence')
        assert presence['faces'] == 2 and presence['visible_users'] == ['A', 'B']
        assert client.health()['perception_ready'] is True
        assert client.request('/debug/perception')['identity_scope'].startswith('visible A/B only')


def test_fixture_uses_publicly_reproducible_character(client):
    module = importlib.import_module('alter_ego.server.app')
    assert module.rt.char_dir is None
    assert client.health()['character'] == '示例角色'
    sprites = client.request('/debug/sprites')
    assert sprites['character'] == '示例角色' and sprites['emotions'] == {}
    with client.websocket() as (ws, _):
        assert client.chat('你好')['ok']
        client.ack(ws, client.receive(ws))
        client.idle()


def test_camera_control_keeps_active_reply_and_off_prevents_idle(client, monkeypatch):
    from alter_ego.server.camera_bridge import CameraMonitor, normalize_snapshot
    module = importlib.import_module('alter_ego.server.app')
    def controller(url, enabled):
        return normalize_snapshot(dict(schema='h04-camera-v1', session_id='switch-test',
            enabled=enabled, ready=enabled, worker_alive=enabled, status='ready',
            frame_age_s=0, face_count=0, visible_users=[]))
    monitor = CameraMonitor(module.rt, 'http://127.0.0.1:9890/snapshot', controller=controller)
    monkeypatch.setattr(module.rt, 'camera_monitor', monitor)
    headers = {'X-Alter-Ego-Camera': '1', 'Origin': client.url}
    with client.websocket() as (ws, hello):
        assert client.chat()['ok']
        speech = client.receive(ws)
        result = client.request('/camera/control', data={'enabled': False}, headers=headers)
        assert result['ok'] and not result['camera']['enabled']
        assert client.receive(ws, 'presence')['camera_enabled'] is False
        assert client.health()['active']['speech_id'] == speech['speech_id']
        client.ack(ws, speech)
        client.idle()
        until(lambda: client.health()['state'] == 'NOTICED')
        assert client.chat()['ok']
        client.ack(ws, client.receive(ws))
        client.idle()
        result = client.request('/camera/control', data={'enabled': True}, headers=headers)
        assert result['ok'] and result['camera']['enabled']
        assert client.receive(ws, 'presence')['camera_enabled'] is True
    with client.websocket() as (ws, hello):
        assert client.receive(ws, 'presence')['camera_enabled'] is True


def test_camera_control_rejects_invalid_requests_and_reports_failure(client, monkeypatch):
    import urllib.error
    module = importlib.import_module('alter_ego.server.app')
    for headers, data, code in [({}, {'enabled': False}, 403),
            ({'X-Alter-Ego-Camera': '1', 'Origin': 'http://other'}, {'enabled': False}, 403),
            ({'X-Alter-Ego-Camera': '1'}, {'enabled': 'false'}, 400),
            ({'X-Alter-Ego-Camera': '1'}, {'enabled': False}, 409)]:
        with pytest.raises(urllib.error.HTTPError) as error:
            client.request('/camera/control', data=data, headers=headers)
        assert error.value.code == code
    class Failure:
        async def set_enabled(self, enabled): raise RuntimeError('mock control failure')
    monkeypatch.setattr(module.rt, 'camera_monitor', Failure())
    assert client.request('/camera/control', data={'enabled': False},
        headers={'X-Alter-Ego-Camera': '1'}) == {'ok': False, 'error': 'camera_control_failed'}


def test_normal_owner_observer_old_and_duplicate_ack(client):
    assert client.request('/debug/users') == {'users': [], 'threshold': None, 'fixture': True}
    with client.websocket() as (owner, hello), client.websocket() as (observer, other):
        assert hello['playback_owner'] and not other['playback_owner']
        result = client.chat()
        speech = client.receive(owner)
        assert result['ok'] and result['status'] == 'playback'
        assert speech['playback_client_id'] == hello['client_id']
        assert base64.b64decode(speech['audio'].split(',', 1)[1]).startswith(b'RIFF')
        client.ack(observer, speech, client_id=hello['client_id'])
        owner.send(json.dumps({'type': 'audio_ended', 'speech_id': 'old', 'turn_id': 'old', 'result': 'ended'}))
        assert client.chat()['status'] == 'busy'
        owner.send(json.dumps({'type': 'audio_started', 'speech_id': speech['speech_id'], 'turn_id': speech['turn_id']}))
        client.ack(owner, speech)
        client.idle()
        second = client.chat('next')
        fresh = client.receive(owner)
        assert second['ok'] and fresh['speech_id'] != speech['speech_id']
        client.ack(owner, speech)
        assert client.chat()['status'] == 'busy'
        client.ack(owner, fresh)
        client.idle()
        rows = client.request('/debug/events')['rows']
        assert all(row['device_id']=='pc-integration-test' for row in rows)
        assert client.health()['device_id']=='pc-integration-test'
        sample = next(s for s in summarize(rows) if s['turn_id'] == speech['turn_id'])
        assert sample['complete'] and sample['tts_sec'] is not None
        assert sample['cloud_sec'] is not None and sample['rewrite_sec'] is not None
        assert sample['stop_to_first_audio_sec'] is None
        assert next(r for r in rows if r['event'] == 'audio_started')['data']['backend'] == 'ws_receipt'
        assert not any('text' in r['data'] for r in rows)


@pytest.mark.parametrize('other_path', ['/debug/chat', '/debug/say'])
def test_concurrent_input_rejected_and_late_thread_quarantined(client, other_path):
    with client.websocket() as (ws, hello):
        client.control(block='cloud')
        pending = client.pool.submit(client.chat)
        until(lambda: client.request('/debug/fixture/control')['started']['cloud'])
        assert client.chat('こんにちは。', other_path)['status'] == 'busy'
        outcome = pending.result(timeout=3)
        assert outcome['status'] == 'timeout' and not outcome['ok']
        assert client.health()['active']['phase'] == 'quarantine'
        assert client.chat()['status'] == 'busy'
        client.control(release='cloud')
        client.idle()
        assert client.request('/debug/fixture/control')['calls']['tts'] == 0
        assert client.chat('recover')['ok']
        speech = client.receive(ws)
        assert speech['speech_id'] != outcome['speech_id']
        client.ack(ws, speech)
        client.idle()
        assert client.request('/debug/fixture/control')['max_concurrent'] == 1


def test_no_frontend_does_not_wedge(client):
    assert client.chat()['status'] == 'no_client'
    assert client.chat('こんにちは。', '/debug/say')['status'] == 'no_client'
    assert client.health()['active'] is None
    with client.websocket() as (ws, hello):
        assert client.chat()['ok']
        client.ack(ws, client.receive(ws))
        client.idle()


@pytest.mark.parametrize('failure', ['cloud', 'json', 'tts'])
def test_failure_degrades_then_recovers(client, failure):
    with client.websocket() as (ws, hello):
        if failure == 'cloud': client.control(cloud_error=True)
        elif failure == 'json': client.control(response='{"zh":"unrelated"}')
        else: client.control(tts_error=True)
        result = client.chat()
        speech = client.receive(ws)
        assert result['status'] == 'degraded'
        if failure != 'tts':
            assert result['subtitle_zh'] == '' and speech['text'] == speech['text_ja']
        else:
            assert speech['audio'] is None
            assert client.health()['components']['tts']['status'] == 'failed'
        client.ack(ws, speech, result='no_audio' if failure == 'tts' else 'ended')
        client.idle()
        samples=summarize(client.request('/debug/events')['rows'])
        sample=next(s for s in samples if s['turn_id']==result['turn_id'])
        if failure!='tts': assert sample['degraded'] and sample['playback_complete']
        client.control(cloud_error=False, tts_error=False,
                       response='{"ja":"はい。","zh":"是的。","emotion":"neutral"}')
        assert client.chat()['status'] == 'playback'
        client.ack(ws, client.receive(ws))
        client.idle()


def test_late_tts_has_no_overlap_and_no_stale_audio(client):
    with client.websocket() as (ws, hello):
        client.control(block='tts')
        pending = client.pool.submit(client.chat)
        until(lambda: client.request('/debug/fixture/control')['started']['tts'])
        expired = pending.result(timeout=3)
        assert expired['status'] == 'timeout'
        assert client.chat()['status'] == 'busy'
        client.control(release='tts')
        client.idle()
        assert client.chat()['ok']
        speech = client.receive(ws)
        assert speech['speech_id'] != expired['speech_id']
        client.ack(ws, speech, result='failed')
        client.idle()
        assert client.request('/debug/fixture/control')['max_concurrent'] == 1


@pytest.mark.parametrize('trigger', ['LEFT', 'timeout', 'IDENTIFIED', 'COUNT_CHANGED'])
def test_stop_ack_precedes_gate_release(client, trigger):
    with client.websocket() as (ws, hello):
        assert client.chat()['ok']
        speech = client.receive(ws)
        if trigger == 'timeout': client.request('/debug/fixture/expire', post=True)
        else: client.request('/debug/fixture/perception', data={'kind': trigger, 'user_id': 'B', 'name': 'same'})
        stop = client.receive(ws, 'stop')
        assert stop['speech_id'] == speech['speech_id']
        assert client.health()['active']['phase'] == 'stopping'
        assert not client.health()['policy']['voice_submit']
        assert client.chat()['status'] == 'busy'
        client.ack(ws, speech, result='cancelled')
        client.idle()
        assert client.health()['policy']['voice_submit']
        assert client.chat()['ok']
        client.ack(ws, client.receive(ws))
        client.idle()


def test_stop_ack_timeout_closes_owner_and_fresh_connection_recovers(client):
    with client.websocket() as (ws, hello):
        client.chat()
        client.receive(ws)
        client.request('/debug/fixture/expire', post=True)
        client.receive(ws, 'stop')
        client.idle()
        assert client.health()['clients'] == 0
    with client.websocket() as (ws, hello):
        assert hello['playback_owner'] and client.chat()['ok']
        client.ack(ws, client.receive(ws))
        client.idle()


def test_identity_change_during_cloud_discards_result_and_preserves_sessions(client):
    with client.websocket() as (ws, hello):
        client.request('/debug/fixture/perception', data={'kind': 'IDENTIFIED', 'user_id': 'A', 'name': 'same'})
        client.chat('A secret')
        client.ack(ws, client.receive(ws))
        client.idle()
        client.control(block='cloud')
        pending = client.pool.submit(client.chat, 'A late')
        until(lambda: client.request('/debug/fixture/control')['started']['cloud'])
        client.request('/debug/fixture/perception', data={'kind': 'IDENTIFIED', 'user_id': 'B', 'name': 'same'})
        client.control(release='cloud')
        assert pending.result(timeout=3)['status'] == 'discarded'
        client.idle()
        result = client.chat('B input')
        assert result['history_turns'] == 1
        module = importlib.import_module('alter_ego.server.app')
        assert 'A secret' not in str(module.rt.fixture_control.messages[-1])
        client.ack(ws, client.receive(ws))
        client.idle()
        client.request('/debug/fixture/perception', data={'kind': 'IDENTIFIED', 'user_id': 'A', 'name': 'renamed'})
        assert client.chat('A again')['history_turns'] == 3
        assert 'B input' not in str(module.rt.fixture_control.messages[-1])
        client.ack(ws, client.receive(ws))
        client.idle()


@pytest.mark.parametrize('asr_result', ['', ['partial', False], ['recognized fixture', True]])
def test_asr_pcm_uses_same_reservation_and_tuple_contract(client, asr_result):
    module = importlib.import_module('alter_ego.server.app')
    with client.websocket() as (ws, hello):
        client.control(asr_result=asr_result)
        pcm = struct.pack('<320h', *([1000]*320))*6 + bytes(640)*21
        assert client.request('/debug/pcm', data=pcm)['status'] == 'accepted_pcm'
        until(lambda: module.rt.lifecycle.last_voice is not None)
        if asr_result == ['recognized fixture', True]:
            speech = client.receive(ws)
            assert client.chat()['status'] == 'busy'
            assert client.request('/debug/pcm', data=pcm)['status'] == 'busy'
            client.ack(ws, speech)
        else:
            assert module.rt.lifecycle.last_voice['status'] == 'empty'
            assert client.request('/debug/fixture/control')['calls']['cloud'] == 0
        client.idle()
        assert client.request('/debug/fixture/control')['calls']['asr'] == 1


def test_asr_inflight_blocks_manual_and_shutdown_waits_for_real_work(client):
    client.control(block='asr')
    pcm = struct.pack('<320h', *([1000]*320))*6 + bytes(640)*21
    client.request('/debug/pcm', data=pcm)
    until(lambda: client.request('/debug/fixture/control')['started']['asr'])
    assert client.chat()['status'] == 'busy'
    client.server.should_exit = True
    until(lambda: importlib.import_module('alter_ego.server.app').rt.lifecycle.closed)
    assert client.server.started
    module = importlib.import_module('alter_ego.server.app')
    assert not module.rt.lifecycle.input.task.done()
    module.rt.fixture_control.releases['asr'].set()


@pytest.mark.parametrize('recognized', ['', ['partial', False], ['preview text', True]])
def test_browser_preview_never_calls_cloud_or_tts_and_commit_is_separate(client, recognized):
    with client.websocket() as (ws, hello):
        client.control(asr_result=recognized)
        # No trailing silence is necessary: the browser supplies a finished sentence.
        pcm = struct.pack('<320h', *([1000] * 320)) * 10
        result = client.request('/debug/transcribe', data=pcm)
        assert result['ok'] and result['status'] == 'asr_preview'
        assert result['text'] == ('preview text' if recognized == ['preview text', True] else '')
        assert client.health()['active'] is None
        calls = client.request('/debug/fixture/control')['calls']
        assert calls['asr'] == 1 and calls['cloud'] == 0 and calls['tts'] == 0
        if result['text']:
            assert client.chat(result['text'])['ok']
            client.ack(ws, client.receive(ws))
            client.idle()
            assert client.request('/debug/fixture/control')['calls']['cloud'] == 1


def test_preview_timeout_quarantines_native_asr_and_rejects_parallel_requests(client):
    client.control(block='asr')
    pcm = struct.pack('<320h', *([1000] * 320)) * 10
    job = client.pool.submit(client.request, '/debug/transcribe', data=pcm)
    until(lambda: client.request('/debug/fixture/control')['started']['asr'])
    assert client.request('/debug/transcribe', data=pcm)['status'] == 'busy'
    assert client.chat()['status'] == 'busy'
    assert job.result(3)['status'] == 'timeout'
    assert client.request('/debug/transcribe', data=pcm)['status'] == 'busy'
    client.control(release='asr')
    client.idle()
    assert client.request('/debug/fixture/control')['calls']['cloud'] == 0


@pytest.mark.parametrize('pcm,code', [(b'', 422), (b'\x01', 422), (bytes(960002), 413)],
                         ids=['empty', 'odd-byte', 'over-limit'])
def test_browser_preview_rejects_invalid_pcm_without_inference(client, pcm, code):
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as error:
        client.request('/debug/transcribe', data=pcm)
    assert error.value.code == code
    assert client.request('/debug/fixture/control')['calls']['asr'] == 0


def test_disconnect_promotes_observer_and_drops_old_completion(client):
    with client.websocket() as (owner, first), client.websocket() as (observer, second):
        client.chat()
        speech = client.receive(owner)
        client.receive(observer)
        owner.close()
        client.idle()
        hello = client.receive(observer, 'hello')
        assert hello['playback_owner']
        client.ack(observer, speech)
        assert client.chat()['ok']
        fresh = client.receive(observer)
        assert fresh['playback_client_id'] == second['client_id']
        client.ack(observer, fresh)
        client.idle()


@pytest.mark.parametrize('enabled', ['0', '1'])
def test_missing_models_start_with_explicit_health(monkeypatch, tmp_path, enabled):
    import asyncio
    module = importlib.import_module('alter_ego.server.app')
    monkeypatch.setattr(module, 'rt', module.Runtime())
    monkeypatch.setenv('ALTER_EGO_FIXTURE', '0')
    monkeypatch.setenv('ALTER_EGO_CHARACTER', str(tmp_path))
    monkeypatch.setenv('ALTER_EGO_PERCEPTION', '0')
    monkeypatch.setenv('ALTER_EGO_TTS', '0')
    monkeypatch.setenv('ALTER_EGO_ASR', enabled)
    monkeypatch.setenv('ALTER_EGO_ASR_MODEL', str(tmp_path/'nonexistent'))
    async def run():
        async with module.lifespan(module.app):
            status = await module.health()
            assert status['components']['asr']['status'] == ('disabled' if enabled=='0' else 'failed')
            assert status['components']['tts']['status'] == 'disabled'
            assert not status['tts_ready'] and module.rt.camera is None
            assert not (await module.debug_chat('input'))['ok']
        assert module.rt.lifecycle.closed
    asyncio.run(run())
