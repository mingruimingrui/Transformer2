"""
Script that contains sacremoses tokenizer as processors
"""

from __future__ import absolute_import
from __future__ import unicode_literals

from ._registry import BaseProcessor, register_processor

__all__ = ['SacremosesTokenize', 'SacremosesDetokenize']


@register_processor('sacremoses_tokenize')
class SacremosesTokenize(BaseProcessor):
    """
    Processor that performs tokenization with the sacremoses library
    Args:
        lang
    """

    def __init__(self, lang='en', custom_nonbreaking_prefixes_file=None):
        try:
            from sacremoses import MosesTokenizer
        except ImportError:
            raise ImportError('Please install package `sacremoses`')
        self.tokenizer = MosesTokenizer(
            lang=lang,
            custom_nonbreaking_prefixes_file=custom_nonbreaking_prefixes_file
        )

    def __call__(self, text):
        return self.tokenizer.tokenize(text, return_str=True)


@register_processor('sacremoses_detokenize')
class SacremosesDetokenize(BaseProcessor):
    """
    Processor that performs detokenization with the sacremoses library
    Args:
        lang
    """

    def __init__(self, lang='en'):
        try:
            from sacremoses import MosesDetokenizer
        except ImportError:
            raise ImportError('Please install package `sacremoses`')
        self.detokenizer = MosesDetokenizer(lang=lang)

    def __call__(self, text):
        return self.detokenizer.detokenize(text.split())
