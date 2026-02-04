"""TTS 同步单次请求 - 发送 1 次 TTS 请求并保存为 WAV"""

import time
from pathlib import Path

from openai import OpenAI

# 单次测试文本（你也可以改成任意一句）
TEST_TEXT = "你好啊，这是第一段测试语音。今天的天气真好，我们一起去郊游吧！"

# 输出目录与文件
OUTPUT_DIR = Path("data-bin/tts-outputs/single-test")
OUTPUT_FILE = OUTPUT_DIR / "speech.wav"


def create_client() -> OpenAI:
    """创建 OpenAI 客户端（指向你的本地 OpenAI-compatible 服务）"""
    return OpenAI(
        base_url="http://192.168.0.213:8000/v1",
        api_key="no-key",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = create_client()

    print("开始同步单次 TTS 请求...")
    print(f"输出文件: {OUTPUT_FILE}")
    print("-" * 60)

    with client.audio.speech.with_streaming_response.create(
        model="fnlp/MOSS-TTSD-v0.5",
        voice="fnlp/MOSS-TTSD-v0.5:anna",
        input=TEST_TEXT,
        response_format="wav",
    ) as response:
        response.stream_to_file(OUTPUT_FILE)


if __name__ == "__main__":
    main()
