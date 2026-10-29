"""Read H04 metadata over loopback; never load face models in the ASR process."""
from __future__ import annotations

import asyncio
import json
import math
import time
import urllib.parse
import urllib.request

SCHEMA = 'h04-camera-v1'


def check_camera_url(value):
    parts = urllib.parse.urlsplit(value)
    if (parts.scheme != 'http' or parts.hostname != '127.0.0.1'
            or not parts.port or parts.username or parts.password or parts.query
            or parts.fragment or parts.path != '/snapshot'):
        raise ValueError('camera_requires_explicit_loopback_snapshot_url')
    return value


def normalize_snapshot(data):
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise ValueError('invalid_camera_schema')
    session = data.get('session_id')
    if not isinstance(session, str) or not 1 <= len(session) <= 64:
        raise ValueError('invalid_camera_session')
    enabled = data.get('enabled', True)
    if type(enabled) is not bool:
        raise ValueError('invalid_camera_enabled')
    ready = enabled and data.get('ready') is True and data.get('worker_alive') is True
    age = data.get('frame_age_s')
    ready = ready and type(age) in (int, float) and math.isfinite(age) and 0 <= age <= 5
    count, users = data.get('face_count'), data.get('visible_users')
    if ready:
        if type(count) is not int or not 0 <= count <= 100:
            raise ValueError('invalid_face_count')
        if (not isinstance(users, list) or len(users) > 2
                or any(u not in ('A', 'B') for u in users)
                or len(set(users)) != len(users) or len(users) > count):
            raise ValueError('invalid_visible_users')
    status = data.get('status')
    if status not in ('loading', 'ready', 'stale', 'failed', 'stopped', 'disabled'):
        status = 'failed'
    return {'schema': SCHEMA, 'session_id': session, 'available': bool(ready), 'enabled': enabled,
            'status': 'disabled' if not enabled else ('ready' if ready else ('stale' if status == 'ready' else status)),
            'face_count': count if ready else 0,
            'visible_users': sorted(users) if ready else [],
            'frame_age_s': round(age, 3) if type(age) in (int, float) and math.isfinite(age) else None}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_snapshot(url):
    check_camera_url(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(url, timeout=1) as response:
        payload = response.read(8193)
    if len(payload) > 8192:
        raise ValueError('camera_snapshot_too_large')
    return normalize_snapshot(json.loads(payload))


def control_camera(url, enabled):
    check_camera_url(url)
    request = urllib.request.Request(url.rsplit('/', 1)[0] + '/control',
        data=json.dumps({'enabled': enabled}).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'X-Alter-Ego-Camera': '1'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=20) as response:
        payload = response.read(8193)
    if len(payload) > 8192:
        raise ValueError('camera_snapshot_too_large')
    view = normalize_snapshot(json.loads(payload))
    if view['enabled'] != enabled:
        raise ValueError('camera_control_not_confirmed')
    return view


def presence_message(view):
    return {'type': 'presence', 'value': view['available'] and view['face_count'] > 0,
            'faces': view['face_count'], 'visible_users': view['visible_users'],
            'available': view['available'], 'camera_status': view['status'],
            'camera_enabled': view.get('enabled', True), 'camera_controllable': True}


class CameraMonitor:
    """Display identities only. A/B are NOT bound to dialogue memory or speaker ID.

    Presence drives wake/sleep after fifteen seconds of confirmed absence.
    A lost camera clears the marker but never fabricates a physical LEFT event.
    """
    def __init__(self, runtime, url, *, reader=read_snapshot, controller=control_camera, clock=time.monotonic):
        self.rt, self.url, self.reader, self.clock = runtime, check_camera_url(url), reader, clock
        self.signature = None
        self.absent_since = None
        self.present = True  # Initial/re-enabled capture also receives the absence grace period.
        self.enabled = True
        self.controller = controller
        self.lock = asyncio.Lock()

    async def apply(self, view):
        rt = self.rt
        enabled = view.get('enabled', True)
        if not enabled or not self.enabled:
            self.present = True
            self.absent_since = None
        self.enabled = enabled
        rt.camera_view = view
        rt.perception_error = '' if view['available'] or not enabled else 'camera_' + view['status']
        if rt.lifecycle:
            rt.lifecycle.status['perception'] = {
                'status': 'disabled' if not enabled else ('ready' if view['available'] else 'degraded'),
                'reason': rt.perception_error}
            if self.present and not rt.lifecycle.coordinator.present:
                rt.lifecycle.coordinator.present = True
                await rt.lifecycle.policy()
        if view['available']:
            rt.visual_hint = '画面中可见{}张人脸。'.format(view['face_count'])
            if view['face_count']:
                self.absent_since = None
                if not self.present:
                    self.present = True
                    if rt.lifecycle:
                        rt.lifecycle.coordinator.present = True
                        await rt.lifecycle.policy()
            else:
                if self.absent_since is None:
                    self.absent_since = self.clock()
                if self.present and self.clock() - self.absent_since >= 15:
                    self.present = False
                    if rt.lifecycle:
                        await rt.lifecycle.leave()
                        await rt.lifecycle.policy()
        else:
            rt.visual_hint = None
            self.absent_since = None
        signature = (view['session_id'], enabled, view['available'], view['status'],
                     view['face_count'], tuple(view['visible_users']))
        if signature != self.signature:
            self.signature = signature
            await rt.hub.send(presence_message(view))

    async def poll_once(self):
        async with self.lock:
            try:
                view = await asyncio.to_thread(self.reader, self.url)
            except Exception:
                view = {'schema': SCHEMA, 'session_id': 'unreachable', 'available': False,
                        'enabled': self.enabled, 'status': 'failed' if self.enabled else 'disabled',
                        'face_count': 0, 'visible_users': [], 'frame_age_s': None}
            if not self.rt.perception_stopping:
                await self.apply(view)

    async def set_enabled(self, enabled):
        async with self.lock:
            view = await asyncio.to_thread(self.controller, self.url, enabled)
            if not self.rt.perception_stopping:
                await self.apply(view)
            return view

    async def run(self):
        while not self.rt.perception_stopping:
            await self.poll_once()
            await asyncio.sleep(.5)
