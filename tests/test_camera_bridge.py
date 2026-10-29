"""Camera metadata integration, without models/devices/private configuration."""
import asyncio
import importlib.util
import json
from pathlib import Path
import threading
import io
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from alter_ego.server.camera_bridge import CameraMonitor, check_camera_url, normalize_snapshot, read_snapshot, control_camera

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('h04_http_test', ROOT / '联调测试/20261011-摄像头集成/camera_http_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def raw(count=1, users=None):
    return dict(schema='h04-camera-v1', session_id='test-session', ready=True,
                worker_alive=True, status='ready', frame_age_s=.1, face_count=count,
                visible_users=['A'] if users is None else users)


@pytest.mark.parametrize('url', ['http://remote/snapshot', 'http://127.0.0.1/snapshot',
    'http://key@127.0.0.1:9890/snapshot', 'http://127.0.0.1:9890/snapshot?key=x',
    'http://127.0.0.1:9890/tts', 'file:///snapshot'])
def test_restrict_loopback(url):
    with pytest.raises(ValueError):
        check_camera_url(url)


def test_reject_corrupt_identity_and_clear_stale_metadata():
    for payload in [raw(count=True), raw(users=['A', 'A']), raw(users=['C']),
                    raw(count=1, users=['A', 'B'])]:
        with pytest.raises(ValueError):
            normalize_snapshot(payload)
    data = raw()
    data['frame_age_s'] = 6
    view = normalize_snapshot(data)
    assert not view['available'] and view['face_count'] == 0 and view['visible_users'] == []


def test_mixed_stdout_through_real_http_and_staleness(monkeypatch):
    now = [10.]
    state = bridge.Snapshot(clock=lambda: now[0])
    state.worker = SimpleNamespace(poll=lambda: None)
    state.consume('Report: some/path')
    state.consume('{bad json')
    assert not state.read()['ready']
    state.consume(json.dumps(dict(event='view', face_count=2, visible_users=['B', 'A'],
                                  frame_arrival_age_sec=.1, inference_sec=.2)))
    server = bridge.make_server(state, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv('http_proxy', 'http://127.0.0.1:1')
        url = 'http://127.0.0.1:{}/snapshot'.format(server.server_address[1])
        view = read_snapshot(url)
        assert view['available'] and view['visible_users'] == ['A', 'B']
        assert view['frame_age_s'] == .3
        now[0] = 16
        view = read_snapshot(url)
        assert not view['available'] and view['face_count'] == 0
        state.worker = SimpleNamespace(poll=lambda: 1)
        assert read_snapshot(url)['status'] == 'failed'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_presence_debounce_fault_and_no_speaker_binding():
    async def scenario():
        now, messages, leaves = [10.], [], []
        async def send(message): messages.append(message)
        async def policy(): pass
        async def leave():
            leaves.append('left')
            life.coordinator.present = False
        life = SimpleNamespace(status={}, coordinator=SimpleNamespace(present=False), policy=policy, leave=leave)
        rt = SimpleNamespace(lifecycle=life, hub=SimpleNamespace(send=send),
                             current_user=None, current_user_id=None, perception_stopping=False)
        monitor = CameraMonitor(rt, 'http://127.0.0.1:9890/snapshot', clock=lambda: now[0])
        await monitor.apply(normalize_snapshot(raw(count=2, users=['A', 'B'])))
        assert life.coordinator.present and rt.current_user is None and rt.current_user_id is None
        assert messages[-1]['visible_users'] == ['A', 'B']
        await monitor.apply(normalize_snapshot(raw(count=1, users=['B'])))
        assert not leaves
        lost = raw()
        lost.update(ready=False, status='failed')
        await monitor.apply(normalize_snapshot(lost))
        assert not leaves and rt.visual_hint is None
        zero = normalize_snapshot(raw(count=0, users=[]))
        await monitor.apply(zero)
        now[0] += 14.9
        await monitor.apply(zero)
        assert not leaves
        now[0] += .1
        await monitor.apply(zero)
        assert leaves == ['left'] and not life.coordinator.present
        await monitor.apply(zero)
        assert leaves == ['left']
        await monitor.apply(normalize_snapshot(raw()))
        assert life.coordinator.present
    asyncio.run(scenario())


def test_disable_wakes_sleeping_character_and_reenable_restarts_15_second_grace():
    async def scenario():
        now, leaves = [0.], []
        async def send(msg): pass
        async def policy(): pass
        async def leave():
            leaves.append(now[0])
            life.coordinator.present = False
        life = SimpleNamespace(status={}, coordinator=SimpleNamespace(present=False), policy=policy, leave=leave)
        rt = SimpleNamespace(lifecycle=life, hub=SimpleNamespace(send=send), perception_stopping=False)
        monitor = CameraMonitor(rt, 'http://127.0.0.1:9890/snapshot', clock=lambda: now[0])
        zero = normalize_snapshot(raw(count=0, users=[]))
        await monitor.apply(zero)
        assert life.coordinator.present  # Initial no-face frame has the full grace period.
        now[0] = 15
        await monitor.apply(zero)
        assert leaves == [15] and not life.coordinator.present
        off = normalize_snapshot(dict(raw(), enabled=False))
        await monitor.apply(off)
        assert life.coordinator.present and rt.perception_error == ''
        assert life.status['perception']['status'] == 'disabled'
        now[0] = 500
        await monitor.apply(off)
        assert leaves == [15] and life.coordinator.present
        await monitor.apply(zero)
        now[0] = 514.99
        await monitor.apply(zero)
        assert leaves == [15]
        now[0] = 515
        await monitor.apply(zero)
        assert leaves == [15, 515]
    asyncio.run(scenario())


def test_real_bridge_http_control_releases_worker_and_restarts_fresh_session(monkeypatch):
    workers = []
    class Worker:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO('')
            self.code = None
            workers.append(self)
        def poll(self): return self.code
        def send_signal(self, sig): self.code = 0
        def wait(self, timeout): return self.code
    monkeypatch.setattr(bridge.subprocess, 'Popen', Worker)
    state = bridge.Snapshot()
    control = bridge.WorkerControl(state, ['fake-camera'], io.StringIO(), io.StringIO())
    control.set_enabled(True)
    state.consume(json.dumps(dict(event='view', face_count=1, visible_users=['A'], frame_arrival_age_sec=0, inference_sec=0)))
    old_session = state.session_id
    server = bridge.make_server(state, 0, control)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = 'http://127.0.0.1:{}/snapshot'.format(server.server_address[1])
    try:
        view = control_camera(url, False)
        assert view['status'] == 'disabled' and not view['available'] and view['visible_users'] == []
        assert workers[0].poll() == 0 and state.worker is None
        assert view['session_id'] != old_session
        assert not state.read()['worker_alive']
        control_camera(url, False)  # OFF is safe to repeat.
        view = control_camera(url, True)
        assert view['enabled'] and view['status'] == 'loading' and view['visible_users'] == []
        assert len(workers) == 2
        control_camera(url, True)
        assert len(workers) == 2  # ON must not steal/reopen an already running device.
        endpoint = url.replace('/snapshot', '/control')
        for headers, body, code in [({}, b'{"enabled":false}', 403),
                ({'X-Alter-Ego-Camera': '1', 'Origin': 'http://other'}, b'{"enabled":false}', 403),
                ({'X-Alter-Ego-Camera': '1'}, b'{"enabled":"false"}', 400)]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(endpoint, data=body, headers=headers))
            assert error.value.code == code
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        control.close()
    assert all(w.poll() == 0 for w in workers)


def test_poll_and_control_are_serialized_and_disabled_survives_read_failure():
    async def scenario():
        messages = []
        entered, release = threading.Event(), threading.Event()
        def reader(url):
            entered.set()
            assert release.wait(2)
            return normalize_snapshot(raw())
        def controller(url, enabled):
            assert release.is_set()
            return normalize_snapshot(dict(raw(), enabled=enabled))
        async def send(msg): messages.append(msg)
        rt = SimpleNamespace(lifecycle=None, hub=SimpleNamespace(send=send), perception_stopping=False)
        monitor = CameraMonitor(rt, 'http://127.0.0.1:9890/snapshot', reader=reader, controller=controller)
        polling = asyncio.create_task(monitor.poll_once())
        assert await asyncio.to_thread(entered.wait, 2)
        changing = asyncio.create_task(monitor.set_enabled(False))
        release.set()
        await asyncio.gather(polling, changing)
        assert messages[-1]['camera_enabled'] is False
        def fail(url): raise ConnectionError()
        monitor.reader = fail
        await monitor.poll_once()
        assert messages[-1]['camera_status'] == 'disabled'
        assert rt.perception_error == ''
    asyncio.run(scenario())


def test_unreachable_camera_does_not_escape_into_dialogue():
    async def scenario():
        messages = []
        async def send(msg): messages.append(msg)
        def fail(url): raise ConnectionError('unreachable')
        rt = SimpleNamespace(lifecycle=None, hub=SimpleNamespace(send=send), perception_stopping=False)
        monitor = CameraMonitor(rt, 'http://127.0.0.1:9890/snapshot', reader=fail)
        await monitor.poll_once()
        assert rt.perception_error == 'camera_failed'
        assert messages[-1]['available'] is False
    asyncio.run(scenario())
