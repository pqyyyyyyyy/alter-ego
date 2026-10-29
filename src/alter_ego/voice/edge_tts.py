"""Optional online Japanese TTS for the browser demo (MP3 transport)."""

from __future__ import annotations


class EdgeTTS:
    def __init__(self, voice: str = "ja-JP-NanamiNeural") -> None:
        import edge_tts

        self._edge_tts = edge_tts
        self.voice = voice

    async def synthesize_mp3(self, text: str) -> bytes:
        audio = bytearray()
        async for chunk in self._edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
                if len(audio) > 2 * 1024 * 1024:
                    raise ValueError("TTS audio exceeds the 2 MB browser transport limit")
        if not audio:
            raise ValueError("TTS returned no audio")
        return bytes(audio)
