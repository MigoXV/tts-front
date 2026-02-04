from pathlib import Path

from tts_front.inferencer.tts.text_preprocessor import TextNormalizer


def main():
    input_path = Path("data-bin/text/test01.txt")
    input_text = input_path.read_text(encoding="utf-8")
    normalizer = TextNormalizer()
    cleaned_text = normalizer.clear_text(input_text)
    print(cleaned_text)


if __name__ == "__main__":
    main()
