"""
Processors for the chinese language
"""


from __future__ import absolute_import
from __future__ import unicode_literals

from ._registry import BaseProcessor, register_processor

__all__ = ['ToSimplifiedChinese', 'ToTraditionalChinese', 'Jieba']


@register_processor('to_simplified_chinese')
class ToSimplifiedChinese(BaseProcessor):
    """
    Processor that performs conversion of chinese characters from traditional
    to simplified
    """

    def __init__(self):
        try:
            from hanziconv import HanziConv
        except ImportError:
            raise ImportError('Please install package `hanziconv`')
        self.convert_fn = HanziConv.toSimplified

    def __call__(self, text):
        return self.convert_fn(text)


@register_processor('to_traditional_chinese')
class ToTraditionalChinese(BaseProcessor):
    """
    Processor that performs conversion of chinese characters from simplified
    to traditional
    """

    def __init__(self):
        try:
            from hanziconv import HanziConv
        except ImportError:
            raise ImportError('Please install package `hanziconv`')
        self.convert_fn = HanziConv.toTraditional

    def __call__(self, text):
        return self.convert_fn(text)


@register_processor('jieba')
class Jieba(BaseProcessor):
    """ Processor that performs jieba tokenization """

    def __init__(self, use_hmm=True):
        self.use_hmm = bool(use_hmm)

    def __call__(self, text):
        try:
            import jieba
        except ImportError:
            raise ImportError('Please install package `jieba`')
        return ' '.join(jieba.cut(text, cut_all=False, HMM=self.use_hmm))
