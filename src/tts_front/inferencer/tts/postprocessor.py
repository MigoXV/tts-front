import struct
from collections.abc import AsyncIterator

import numpy as np


class AudioPostProcessor:
    """流式音频后处理：淡入/淡出，减少分段拼接时的哒哒声。"""

    def __init__(self, fade_ms: int = 10, emit_header: bool = True):
        self.fade_ms = max(0, int(fade_ms))
        self.emit_header = emit_header

        self._header_buffer = b""
        self._header_parsed = False
        self._sample_rate = None
        self._channels = None
        self._bits_per_sample = None
        self._frame_size = None
        self._fade_samples = 0
        self._samples_emitted = 0
        self._pcm_buffer = b""
        self._bypass = False

    def _try_parse_header(self, data: bytes):
        if len(data) < 12:
            return None
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise ValueError("Not a WAV file")

        offset = 12
        fmt = None
        data_offset = None

        while True:
            if len(data) < offset + 8:
                return None

            chunk_id = data[offset : offset + 4]
            chunk_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
            if len(data) < offset + 8 + chunk_size:
                return None

            if chunk_id == b"fmt ":
                fmt = struct.unpack("<HHIIHH", data[offset + 8 : offset + 24])
            elif chunk_id == b"data":
                data_offset = offset + 8
                if fmt is not None:
                    audio_format, channels, sample_rate, _, _, bits_per_sample = fmt
                    return data_offset, sample_rate, channels, bits_per_sample

            offset += 8 + chunk_size

    def _prepare(self, data_offset: int, sample_rate: int, channels: int, bits: int):
        self._header_parsed = True
        self._sample_rate = sample_rate
        self._channels = channels
        self._bits_per_sample = bits
        if bits in (16, 32):
            self._frame_size = (bits // 8) * channels
            if self.fade_ms > 0:
                self._fade_samples = max(1, int(sample_rate * self.fade_ms / 1000))
        else:
            self._bypass = True

        if self.emit_header:
            return self._header_buffer[:data_offset]
        return b""

    def _apply_fade_in(self, frames: np.ndarray) -> np.ndarray:
        if self._fade_samples <= 0:
            return frames
        remaining = self._fade_samples - self._samples_emitted
        if remaining <= 0:
            return frames

        count = min(remaining, frames.shape[0])
        if self._fade_samples == 1:
            gains = np.ones((count, 1), dtype=np.float32)
        else:
            idx = np.arange(self._samples_emitted, self._samples_emitted + count)
            gains = (idx / (self._fade_samples - 1)).astype(np.float32).reshape(-1, 1)

        frames[:count] = frames[:count] * gains
        return frames

    def _apply_fade_out(self, frames: np.ndarray) -> np.ndarray:
        if self._fade_samples <= 0:
            return frames
        count = min(self._fade_samples, frames.shape[0])
        if count <= 0:
            return frames

        if count == 1:
            gains = np.zeros((1, 1), dtype=np.float32)
        else:
            gains = np.linspace(
                1.0, 0.0, count, endpoint=True, dtype=np.float32
            ).reshape(-1, 1)
        frames[-count:] = frames[-count:] * gains
        return frames

    def _process_frames(self, pcm_bytes: bytes, is_final: bool) -> bytes:
        if self._bypass or self._frame_size is None:
            return pcm_bytes

        dtype = np.int16 if self._bits_per_sample == 16 else np.int32
        if not pcm_bytes:
            return b""

        array = np.frombuffer(pcm_bytes, dtype=dtype).copy()
        if self._channels and self._channels > 1:
            frames = array.reshape(-1, self._channels)
        else:
            frames = array.reshape(-1, 1)

        frames = frames.astype(np.float32)
        frames = self._apply_fade_in(frames)
        if is_final:
            frames = self._apply_fade_out(frames)

        self._samples_emitted += frames.shape[0]

        if self._bits_per_sample == 16:
            min_val, max_val = -32768, 32767
        else:
            min_val, max_val = -2147483648, 2147483647
        frames = np.clip(frames, min_val, max_val).astype(dtype)

        return frames.reshape(-1).tobytes()

    async def process_wav_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[bytes]:
        async for chunk in chunks:
            if not self._header_parsed:
                self._header_buffer += chunk
                parsed = self._try_parse_header(self._header_buffer)
                if parsed is None:
                    continue
                data_offset, sample_rate, channels, bits = parsed
                header_bytes = self._prepare(data_offset, sample_rate, channels, bits)
                if header_bytes:
                    yield header_bytes
                pcm_bytes = self._header_buffer[data_offset:]
                self._header_buffer = b""
            else:
                pcm_bytes = chunk

            if self._bypass:
                if pcm_bytes:
                    yield pcm_bytes
                continue

            self._pcm_buffer += pcm_bytes
            if self._frame_size is None:
                continue

            keep_bytes = self._fade_samples * self._frame_size
            if len(self._pcm_buffer) <= keep_bytes:
                continue

            process_len = len(self._pcm_buffer) - keep_bytes
            process_len -= process_len % self._frame_size
            if process_len <= 0:
                continue

            to_process = self._pcm_buffer[:process_len]
            self._pcm_buffer = self._pcm_buffer[process_len:]
            out_bytes = self._process_frames(to_process, is_final=False)
            if out_bytes:
                yield out_bytes

        if self._bypass:
            if self._pcm_buffer:
                yield self._pcm_buffer
            return

        if self._pcm_buffer and self._frame_size:
            process_len = len(self._pcm_buffer) - (
                len(self._pcm_buffer) % self._frame_size
            )
            to_process = self._pcm_buffer[:process_len]
            remainder = self._pcm_buffer[process_len:]
            out_bytes = self._process_frames(to_process, is_final=True)
            if out_bytes:
                yield out_bytes
            if remainder:
                yield remainder
