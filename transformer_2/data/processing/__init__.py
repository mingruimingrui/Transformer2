from __future__ import absolute_import
from __future__ import unicode_literals

# Registry
from transformer_2.data.processing._registry import PROCESSOR_REGISTRY, \
    BaseProcessor, register_processor, Compose, make_processor_from_list

# Generic processors
from transformer_2.data.processing.general import HtmlUnescape, HtmlEscape, \
    Lowercase, Strip, WhitespaceNormalize, UnicodeNormalize
from transformer_2.data.processing.replace import ReplaceSubstrings, \
    ReplaceTokens, WhitelistCharacters

# Specialized processors
from transformer_2.data.processing.onmt import OnmtTokenize, OnmtDetokenize
from transformer_2.data.processing.sentencepiece import SpmEncode, SpmDecode
from transformer_2.data.processing.sacremoses import SacremosesTokenize, \
    SacremosesDetokenize

# Language specific processors
from .chinese import ToSimplifiedChinese, ToTraditionalChinese, Jieba

__all__ = [
    'PROCESSOR_REGISTRY', 'BaseProcessor', 'register_processor',
    'Compose', 'make_processor_from_list',

    'HtmlUnescape', 'HtmlEscape',
    'Lowercase', 'Strip',
    'WhitespaceNormalize', 'UnicodeNormalize',
    'ReplaceSubstrings', 'ReplaceTokens', 'WhitelistCharacters',

    'OnmtTokenize', 'OnmtDetokenize',
    'SacremosesTokenize', 'SacremosesDetokenize',
    'SpmEncode', 'SpmDecode',

    'ToSimplifiedChinese', 'ToTraditionalChinese', 'Jieba'
]
