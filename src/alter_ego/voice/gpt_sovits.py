"""HTTP client for a GPT-SoVITS host; no model dependencies on the caller."""

from __future__ import annotations

import asyncio
import io
import json
import math
import urllib.parse
import urllib.request
import wave


class RemoteGPTSoVITS:
    def __init__(self, url: str, timeout: float = 90) -> None:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("ALTER_EGO_TTS_URL must be an HTTP(S) /tts endpoint")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("TTS timeout must be positive and finite")
        self.url = url
        self.timeout = timeout

    def _request(self, text: str) -> bytes:
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.headers.get_content_type() not in {"audio/wav", "audio/x-wav"}:
                raise ValueError("TTS did not return WAV audio")
            audio = response.read(2 * 1024 * 1024 + 1)
        if len(audio) > 2 * 1024 * 1024:
            raise ValueError("TTS audio exceeds the 2 MB browser transport limit")
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if wav.getnframes() == 0 or not wav.readframes(1):
                raise ValueError("TTS returned empty WAV audio")
        return audio

    async def synthesize_wav(self, text: str) -> bytes:
        return await asyncio.to_thread(self._request, text)

    async def check_ready(self) -> None:
        def check():
            parts = urllib.parse.urlsplit(self.url)
            url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, '/health', '', ''))
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=5) as response:
                state = json.loads(response.read(65536))
            if state.get('ready') is not True or state.get('busy'):
                raise RuntimeError('tts_not_ready')
        await asyncio.to_thread(check)
