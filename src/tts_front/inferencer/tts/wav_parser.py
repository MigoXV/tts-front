import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class WavInfo:
    """WAV file information dataclass"""

    channels: int
    sample_rate: int
    bits_per_sample: int
    audio_format: int
    pcm: bytes

    @property
    def dtype(self) -> np.dtype:
        """Get numpy dtype based on bits_per_sample"""
        if self.bits_per_sample == 8:
            return np.dtype(np.uint8)
        elif self.bits_per_sample == 16:
            return np.dtype(np.int16)
        elif self.bits_per_sample == 24:
            return np.dtype(np.int32)  # 24-bit needs special handling
        elif self.bits_per_sample == 32:
            return np.dtype(np.int32)
        else:
            raise ValueError(f"Unsupported bits_per_sample: {self.bits_per_sample}")

    def to_numpy(self) -> np.ndarray:
        """Convert PCM bytes to numpy array"""
        if self.bits_per_sample == 24:
            # Special handling for 24-bit audio
            # Convert 24-bit to 32-bit by padding
            samples = len(self.pcm) // 3
            array = np.zeros(samples, dtype=np.int32)
            for i in range(samples):
                # Read 3 bytes and convert to int32
                byte_data = self.pcm[i * 3 : (i + 1) * 3]
                # Add padding byte and interpret as int32
                value = int.from_bytes(
                    byte_data + b"\x00", byteorder="little", signed=True
                )
                array[i] = value >> 8  # Shift right to maintain proper scale
            return array
        else:
            # Standard conversion for 8, 16, 32 bit
            array = np.frombuffer(self.pcm, dtype=self.dtype)

        # Reshape to (samples, channels) if stereo
        if self.channels > 1:
            array = array.reshape(-1, self.channels)

        return array

    def to_normalized_numpy(self) -> np.ndarray:
        """Convert PCM to normalized float array in range [-1.0, 1.0]"""
        array = self.to_numpy()

        # Get the maximum value for the data type
        if self.bits_per_sample == 8:
            # 8-bit is unsigned (0-255), convert to signed first
            array = array.astype(np.float32) - 128.0
            max_val = 128.0
        elif self.bits_per_sample == 16:
            array = array.astype(np.float32)
            max_val = 32768.0  # 2^15
        elif self.bits_per_sample == 24:
            array = array.astype(np.float32)
            max_val = 8388608.0  # 2^23
        elif self.bits_per_sample == 32:
            array = array.astype(np.float32)
            max_val = 2147483648.0  # 2^31
        else:
            raise ValueError(f"Unsupported bits_per_sample: {self.bits_per_sample}")

        # Normalize to [-1.0, 1.0]
        return array / max_val


def parse_wav_bytes(data: bytes) -> WavInfo:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV file")

    offset = 12
    fmt = None
    data_chunk = None

    while offset < len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        chunk_data = data[offset + 8 : offset + 8 + chunk_size]

        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", chunk_data[:16])
        elif chunk_id == b"data":
            data_chunk = chunk_data

        offset += 8 + chunk_size

    if not fmt or not data_chunk:
        raise ValueError("Invalid WAV")

    audio_format, channels, sample_rate, _, _, bits_per_sample = fmt

    return WavInfo(
        channels=channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        audio_format=audio_format,
        pcm=data_chunk,
    )
