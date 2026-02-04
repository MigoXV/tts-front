import asyncio
import sounddevice as sd
import numpy as np
import io
import struct
import time
from tts_front.inferencer.tts.inferencer import AsyncInferencer
from tts_front.inferencer.tts.text_preprocessor import SPLIT_TOKEN


async def play_stream_with_latency(
    inferencer: AsyncInferencer,
    text: str,
    model: str,
    voice: str,
    enable_split: bool,
) -> float:
    start_time = time.perf_counter()
    first_chunk_time = None

    print("开始异步流式 TTS 请求并实时播放...")
    print("-" * 60)

    # WAV 文件头部信息
    sample_rate = None
    channels = None
    dtype = None
    bits_per_sample = None
    header_parsed = False

    # 缓冲区
    full_audio = io.BytesIO()
    sample_buffer = b""
    accumulated_header = b""
    data_start_offset = 0

    # 音频数据列表
    audio_chunks = []

    def parse_wav_header(data: bytes):
        """解析 WAV 头部获取音频参数"""
        if len(data) < 44:
            return None

        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return None

        offset = 12
        while offset < len(data) - 8:
            chunk_id = data[offset : offset + 4]
            chunk_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]

            if chunk_id == b"fmt ":
                fmt_data = data[offset + 8 : offset + 8 + chunk_size]
                audio_format, channels, sample_rate, _, _, bits_per_sample = (
                    struct.unpack("<HHIIHH", fmt_data[:16])
                )

                if bits_per_sample == 16:
                    dtype = np.int16
                elif bits_per_sample == 24:
                    dtype = np.int32
                elif bits_per_sample == 32:
                    dtype = np.int32
                else:
                    dtype = np.int16

                return {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "bits_per_sample": bits_per_sample,
                    "dtype": dtype,
                }

            offset += 8 + chunk_size

        return None

    def process_pcm_data(
        pcm_data: bytes, dtype, bits_per_sample: int, channels: int
    ) -> np.ndarray:
        """将PCM数据转换为归一化的numpy数组"""
        audio_array = np.frombuffer(pcm_data, dtype=dtype)

        # 归一化
        if bits_per_sample == 16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif bits_per_sample == 24 or bits_per_sample == 32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0

        # 确保形状正确
        if channels == 1:
            audio_array = audio_array.reshape(-1, 1)
        else:
            audio_array = audio_array.reshape(-1, channels)

        return audio_array

    print("接收音频数据中...")

    async for chunk in inferencer.infer_stream(
        text=text,
        model="miratts",
        voice="rita",
        enable_split=enable_split,
    ):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
        full_audio.write(chunk)

        # 解析头部
        if not header_parsed:
            accumulated_header += chunk

            if len(accumulated_header) >= 44:
                header_info = parse_wav_header(accumulated_header)

                if header_info:
                    sample_rate = header_info["sample_rate"]
                    channels = header_info["channels"]
                    dtype = header_info["dtype"]
                    bits_per_sample = header_info["bits_per_sample"]

                    print(
                        f"音频参数 - 采样率: {sample_rate} Hz, 声道: {channels}, 位深: {bits_per_sample} bit"
                    )
                    print("-" * 60)

                    # 找到 data chunk 开始位置
                    offset = 12
                    while offset < len(accumulated_header) - 8:
                        chunk_id = accumulated_header[offset : offset + 4]
                        chunk_size = struct.unpack(
                            "<I", accumulated_header[offset + 4 : offset + 8]
                        )[0]

                        if chunk_id == b"data":
                            data_start_offset = offset + 8
                            break

                        offset += 8 + chunk_size

                    header_parsed = True

                    # 处理头部后的剩余数据
                    if len(accumulated_header) > data_start_offset:
                        sample_buffer = accumulated_header[data_start_offset:]
        else:
            # 头部已解析，累积数据到缓冲区
            sample_buffer += chunk

        # 处理缓冲区中的完整采样点
        if header_parsed and len(sample_buffer) > 0:
            bytes_per_sample = (bits_per_sample // 8) * channels

            # 只处理完整的采样点
            complete_samples = len(sample_buffer) // bytes_per_sample
            if complete_samples > 0:
                bytes_to_process = complete_samples * bytes_per_sample
                pcm_data = sample_buffer[:bytes_to_process]
                sample_buffer = sample_buffer[bytes_to_process:]

                audio_array = process_pcm_data(
                    pcm_data, dtype, bits_per_sample, channels
                )
                audio_chunks.append(audio_array)

    print(f"数据接收完成，总共接收: {full_audio.tell()} 字节")
    print("开始播放...")

    # 合并所有音频数据并播放
    if audio_chunks:
        full_audio_array = np.vstack(audio_chunks)
        sd.play(full_audio_array, samplerate=sample_rate)
        sd.wait()

    print("播放完成！")
    if first_chunk_time is None:
        return -1.0
    return first_chunk_time - start_time


async def main():
    inferencer = AsyncInferencer()
    # 20 段长文本 + 切割标记（SPLIT_TOKEN），用于更明显地观察“分句并发”效果
    test_text = SPLIT_TOKEN.join(
        [
            f"第{i:02d}段：这是一段用于测试的朗读文本。内容包含数字{i}，以及简单的停顿标点，便于观察流式播放与首包延迟。"
            for i in range(1, 21)
        ]
    )

    print("不启用分句：测试首包延迟")
    latency_no_split = await play_stream_with_latency(
        inferencer=inferencer,
        text=test_text,
        model="miratts",
        voice="rita",
        enable_split=False,
    )
    print(f"首包延迟(不分句): {latency_no_split:.3f} s")
    print("=" * 60)

    print("启用分句：测试首包延迟")
    latency_split = await play_stream_with_latency(
        inferencer=inferencer,
        text=test_text,
        model="miratts",
        voice="rita",
        enable_split=True,
    )
    print(f"首包延迟(分句): {latency_split:.3f} s")


if __name__ == "__main__":
    asyncio.run(main())
