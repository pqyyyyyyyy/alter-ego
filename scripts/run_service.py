"""Own one service process; fence stuck native calls by terminating that process.

No automatic restart: an operator inspects the failure before starting another
instance. External model/HTTP services are outside this process's ownership.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


async def child(port, stop_path):
    import uvicorn
    config = uvicorn.Config('alter_ego.server.app:app', host='127.0.0.1', port=port,
                            log_level='info', ws='websockets-sansio')
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not task.done():
        if Path(stop_path).exists(): server.should_exit = True
        await asyncio.sleep(.1)
    await task


def supervise(port, log_dir, hard_worker_seconds=180., shutdown_seconds=10., stop_file=None):
    root = Path(__file__).resolve().parents[1]
    log_dir.mkdir(parents=True, exist_ok=True)
    if stop_file and stop_file.exists(): raise FileExistsError('stop file already exists; choose a new run directory')
    events = log_dir / 'supervisor.jsonl'
    def record(event, **data):
        with events.open('a', encoding='utf-8') as out:
            out.write(json.dumps({'event': event, 'mono_s': time.monotonic(), **data})+'\n')
    # Refuse an occupied port before creating a child; never monitor an unrelated server.
    probe = socket.socket()
    try: probe.bind(('127.0.0.1', port))
    finally: probe.close()
    with tempfile.TemporaryDirectory(prefix='alter-ego-supervisor-') as temp:
        stop_path = Path(temp) / 'stop'
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root / 'src')
        env['PYTHONUTF8'] = '1'
        env['ALTER_EGO_SUPERVISED'] = '1'
        with (log_dir / 'service.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child',
                                        '--port', str(port), '--stop-path', str(stop_path)],
                                       cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            record('started', pid=process.pid, port=port)
            last_health = time.monotonic()
            started = last_health
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                while process.poll() is None:
                    if stop_file and stop_file.exists():
                        record('shutdown_requested', pid=process.pid)
                        break
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/health', timeout=.5) as response:
                            health = json.load(response)
                        last_health = time.monotonic()
                        active = health.get('active')
                        if active and active.get('phase') == 'quarantine' and active['age_s'] >= hard_worker_seconds:
                            record('worker_deadline', pid=process.pid, phase=active['phase'])
                            process.kill()
                            process.wait(timeout=5)
                            return 75
                    except (OSError, ValueError):
                        if time.monotonic()-max(started, last_health) >= hard_worker_seconds:
                            record('unresponsive_deadline', pid=process.pid)
                            process.kill()
                            process.wait(timeout=5)
                            return 75
                    time.sleep(.1)
            except KeyboardInterrupt:
                record('shutdown_requested', pid=process.pid)
            finally:
                if process.poll() is None:
                    stop_path.touch()
                    try: process.wait(timeout=shutdown_seconds)
                    except subprocess.TimeoutExpired:
                        record('shutdown_deadline', pid=process.pid)
                        process.kill()
                        process.wait(timeout=5)
                record('exited', pid=process.pid, returncode=process.returncode)
            return process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--log-dir', type=Path, default=Path('run-logs'))
    parser.add_argument('--hard-worker-seconds', type=float, default=180.)
    parser.add_argument('--shutdown-seconds', type=float, default=10.)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--stop-path', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535: parser.error('invalid port')
    if min(args.hard_worker_seconds, args.shutdown_seconds) <= 0: parser.error('positive deadlines required')
    if args.fixture: os.environ['ALTER_EGO_FIXTURE'] = '1'
    if args.child:
        asyncio.run(child(args.port, args.stop_path))
        return 0
    return supervise(args.port, args.log_dir, args.hard_worker_seconds, args.shutdown_seconds, args.stop_file)


if __name__ == '__main__':
    raise SystemExit(main())
