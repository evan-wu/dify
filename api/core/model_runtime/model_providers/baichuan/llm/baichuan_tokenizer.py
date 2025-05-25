import re


class BaichuanTokenizer:
    @classmethod
    def count_chinese_characters(cls, text: str) -> int:
        return len(re.findall("[\\u4e00-\\u9fa5]", text))

    @classmethod
    def count_english_vocabularies(cls, text: str) -> int:
        text = re.sub("[^a-zA-Z0-9\\s]", "", text)
        return len(text.split())

    @classmethod
    def _get_num_tokens(cls, text: str) -> int:
        return int(cls.count_chinese_characters(text) + cls.count_english_vocabularies(text) * 1.3)
