"""
Processors to do tokenization with the open nmt tokenizer
"""

from __future__ import absolute_import
from __future__ import unicode_literals

from ._registry import BaseProcessor, register_processor

__all__ = ['OnmtTokenize', 'OnmtDetokenize']


@register_processor('onmt_tokenize')
class OnmtTokenize(BaseProcessor):
    """
    Processor that performs onmt tokenization
    Check https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md
    for options
    """

    def __init__(self, mode='conservative', **kwargs):
        try:
            import pyonmttok
        except ImportError:
            raise ImportError('Please install package `pyonmttok`')
        self.tokenizer = pyonmttok.Tokenizer(mode=mode, **kwargs)

    def __call__(self, text):
        tokens, _ = self.tokenizer.tokenize(text)
        return ' '.join(tokens)


@register_processor('onmt_detokenize')
class OnmtDetokenize(BaseProcessor):
    """
    Processor that performs onmt detokenization
    Check https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md
    for options
    """

    def __init__(self, mode='conservative', **kwargs):
        try:
            import pyonmttok
        except ImportError:
            raise ImportError('Please install package `pyonmttok`')
        self.tokenizer = pyonmttok.Tokenizer(mode=mode, **kwargs)

    def __call__(self, text):
        return self.tokenizer.detokenize(text.split())
