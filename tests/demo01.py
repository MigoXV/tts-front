"""TTS 同步单次请求 - 发送 1 次 TTS 请求并保存为 WAV"""

import time
from pathlib import Path

from openai import OpenAI

# 单次测试文本（你也可以改成任意一句）
TEST_TEXT = """欢迎使用OpenAI TTS接口。这是一个同步单次请求的示例代码，演示如何发送文本并将生成的语音保存为文件。祝你使用愉快！"""

# 输出目录与文件
OUTPUT_DIR = Path("data-bin/tts-outputs/single-test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "speech.wav"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    print("开始同步单次 TTS 请求...")
    print(f"输出文件: {OUTPUT_FILE}")
    print("-" * 60)

    with client.audio.speech.with_streaming_response.create(
        model="miratts",
        voice="rita",
        input=TEST_TEXT,
        response_format="wav",
    ) as response:
        response.stream_to_file(OUTPUT_FILE)


if __name__ == "__main__":
    main()
