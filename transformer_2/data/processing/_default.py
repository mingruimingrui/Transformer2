"""
Location storing default processing steps for each language
"""

DEFAULT_PROCESSING_STEPS = {}


# EN
DEFAULT_PROCESSING_STEPS['en'] = [
    'unicode_normalize',
    'html_unescape',
    'lowercase',
    'whitespace_normalize',
    'onmt_tokenize'
]


# DE
DEFAULT_PROCESSING_STEPS['de'] = [
    'unicode_normalize',
    'html_unescape',
    'lowercase',
    'whitespace_normalize',
    'onmt_tokenize'
]


# FR
DEFAULT_PROCESSING_STEPS['fr'] = [
    'unicode_normalize',
    'html_unescape',
    'lowercase',
    'whitespace_normalize',
    'onmt_tokenize'
]


# ZH
DEFAULT_PROCESSING_STEPS['zh'] = [
    'unicode_normalize',
    'html_unescape',
    'lowercase',
    'to_simplified_chinese',
    'whitespace_normalize',
    'jieba'
]
