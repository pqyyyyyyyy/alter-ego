import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request


def test_native_hang_is_fenced_by_owned_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sock = socket.socket()
    sock.bind(('127.0.0.1',0))
    port = sock.getsockname()[1]
    sock.close()
    env = {**os.environ, 'ALTER_EGO_MODEL_TIMEOUT': '.2', 'PYTHONUTF8': '1'}
    launcher = subprocess.Popen([sys.executable, str(root/'scripts/run_service.py'), '--fixture',
        '--port', str(port), '--log-dir', str(tmp_path), '--hard-worker-seconds', '1.5',
        '--stop-file',str(tmp_path/'stop')],
        cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f'http://127.0.0.1:{port}'
    deadline = time.monotonic()+6
    try:
        while True:
            try:
                with opener.open(url+'/health', timeout=.2) as response:
                    assert json.load(response)['supervised']
                break
            except OSError:
                assert time.monotonic()<deadline, 'supervised service failed to start'
                time.sleep(.02)
        req = urllib.request.Request(url+'/debug/fixture/control', data=b'{"block":"cloud"}',
                                     headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=2): pass
        req = urllib.request.Request(url+'/debug/chat?text=fixture', data=b'')
        with opener.open(req,timeout=2) as response:
            assert json.load(response)['status']=='timeout'
        assert launcher.wait(timeout=6)==75
        events = [json.loads(row) for row in (tmp_path/'supervisor.jsonl').read_text().splitlines()]
        assert any(row['event']=='worker_deadline' for row in events)
        assert events[-1]['event']=='exited'
        probe = socket.socket()
        try: assert probe.connect_ex(('127.0.0.1',port))!=0
        finally: probe.close()
    finally:
        if launcher.poll() is None:
            (tmp_path/'stop').touch()
            launcher.wait(timeout=15)
