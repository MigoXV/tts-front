import asyncio
from collections.abc import AsyncIterator
from typing import Dict, Tuple

import numpy as np
import openai

from .llm_prompt import (LLM_NORMALIZE_SYSTEM_PROMPT,
                         LLM_NORMALIZE_USER_TEMPLATE)
from .postprocessor import AudioPostProcessor
from .text_preprocessor import TextNormalizer
from .wav_parser import parse_wav_bytes


class AsyncInferencer:
    """异步推理器"""

    def __init__(self, voice_map: Dict[str, str] = {}):
        self.client = openai.AsyncOpenAI()
        self.normalizer = TextNormalizer()
        self.voice_map = voice_map

    async def infer(self, text: str, model: str, voice: str) -> Tuple[np.ndarray, int]:
        text = self.normalizer.normalize(text)
        audio_data = await self.client.audio.speech.create(
            model=model,
            voice=self.voice_map.get(voice, voice),
            input=text,
            response_format="wav",
        )
        audio_data = await audio_data.aread()
        wav_info = parse_wav_bytes(audio_data)
        return wav_info.to_normalized_numpy(), wav_info.sample_rate

    async def infer_stream(
        self,
        text: str,
        model: str,
        voice: str,
        enable_split: bool = False,
    ) -> AsyncIterator[bytes]:
        if not enable_split:
            text = self.normalizer.normalize(text)
            async with self.client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                response_format="wav",
            ) as response:
                async for chunk in response.iter_bytes():
                    yield chunk
            return

        segments = [
            self.normalizer.normalize(s) for s in self.normalizer.split_text(text)
        ]
        segments = [s for s in segments if s]
        if not segments:
            return

        semaphore = asyncio.Semaphore(4)

        async def stream_segment(
            segment: str,
            queue: asyncio.Queue[bytes | None],
            emit_header: bool,
        ) -> None:
            async with semaphore:
                async with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=self.voice_map.get(voice, voice),
                    input=segment,
                    response_format="wav",
                ) as response:
                    postprocessor = AudioPostProcessor(emit_header=emit_header)
                    async for chunk in postprocessor.process_wav_stream(
                        response.iter_bytes()
                    ):
                        await queue.put(chunk)
            await queue.put(None)

        async with self.client.audio.speech.with_streaming_response.create(
            model=model,
            voice=self.voice_map.get(voice, voice),
            input=segments[0],
            response_format="wav",
        ) as response:
            postprocessor = AudioPostProcessor(emit_header=True)
            async for chunk in postprocessor.process_wav_stream(response.iter_bytes()):
                yield chunk

        if len(segments) == 1:
            return

        queues = [asyncio.Queue() for _ in segments[1:]]
        tasks = [
            asyncio.create_task(stream_segment(segment, queue, False))
            for segment, queue in zip(segments[1:], queues)
        ]

        try:
            for queue in queues:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
        finally:
            for task in tasks:
                task.cancel()

    async def llm_normalize(self, text: str, model: str) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_NORMALIZE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": LLM_NORMALIZE_USER_TEMPLATE.format(text=text),
                },
            ],
            temperature=0,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
