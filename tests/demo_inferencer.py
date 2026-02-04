import asyncio
import soundfile as sf

from tts_front.inferencer.tts.inferencer import AsyncInferencer


async def main():
    inferencer = AsyncInferencer()
    test_text = """欢迎使用OpenAI TTS接口。这是一个异步单次请求的示例代码，演示如何发送文本并将生成的语音保存为文件。祝你使用愉快！"""
    output_file = "data-bin/tts-outputs/single-test/speech.wav"

    print("开始异步单次 TTS 请求...")
    print(f"输出文件: {output_file}")
    print("-" * 60)

    audio, sr = await inferencer.infer(
        text=test_text,
        model="miratts",
        voice="rita",
    )
    sf.write(output_file, audio, sr)
    print("完成！")


if __name__ == "__main__":
    asyncio.run(main())
