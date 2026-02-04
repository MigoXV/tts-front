import re


class TextNormalizer:
    SPLIT_TOKEN = "<TTS SPLIT>"
    SPECIAL_CHARS = ["\n", "\r", "\t", "  ", "`", "**"]
    _TN_CACHE = {}

    def __init__(self, lang: str = "zh"):
        self.lang = lang

    def split_text(self, text: str):
        """根据 SPLIT_TOKEN 分割文本"""
        return [
            segment.strip()
            for segment in text.split(self.SPLIT_TOKEN)
            if segment.strip()
        ]

    def clear_emojis(self, text: str):
        """移除文本中的表情符号"""
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # 表情符号
            "\U0001f300-\U0001f5ff"  # 符号与图标
            "\U0001f680-\U0001f6ff"  # 交通与地图符号
            "\U0001f1e0-\U0001f1ff"  # 国旗
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    def clear_special_chars(self, text: str):
        """清理文本中的换行符等特殊字符"""
        for char in self.SPECIAL_CHARS:
            text = text.replace(char, " ")
        # 合并多个空格为一个空格
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clear_text(self, text: str):
        """综合清理文本"""
        # text = self.clear_emojis(text)
        text = self.clear_special_chars(text)
        return text

    def tn_text(self, text: str) -> str:
        """Text normalization (TN) via WeTextProcessing."""
        if not text:
            return text

        normalizer = self._TN_CACHE.get(self.lang)
        if normalizer is None:
            if self.lang == "zh":
                from tn.chinese.normalizer import Normalizer
            elif self.lang == "en":
                from tn.english.normalizer import Normalizer
            else:
                raise ValueError(f"Unsupported lang: {self.lang}")
            normalizer = Normalizer()
            self._TN_CACHE[self.lang] = normalizer

        return normalizer.normalize(text)

    def normalize(self, text: str) -> str:
        """TTS 朗读前统一规范化入口。"""
        text = self.clear_text(text)
        text = self.tn_text(text)
        return text


SPLIT_TOKEN = TextNormalizer.SPLIT_TOKEN
