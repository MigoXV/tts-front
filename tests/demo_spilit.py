from tts_front.inferencer.tts.text_preprocessor import TextNormalizer


def main():
    test_text = (
        """这是第一段文本。<TTS SPLIT>这是第二段文本。<TTS SPLIT>这是第三段文本。"""
    )
    normalizer = TextNormalizer()
    segments = normalizer.split_text(test_text)
    for i, segment in enumerate(segments):
        print(f"Segment {i + 1}: {segment}")


if __name__ == "__main__":
    main()
