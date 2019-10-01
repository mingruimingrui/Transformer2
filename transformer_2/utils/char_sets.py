# -*- coding: utf-8 -*-

"""
Script containing some commonly used character sets
"""

from __future__ import absolute_import
from __future__ import unicode_literals

import string
import itertools

__all__ = [
    'VALID_CHAR_SETS',
    'en_valid_char_set',
    'de_valid_char_set',
    'zh_valid_char_set',
    'th_valid_char_set',
    'vn_valid_char_set'
]

VALID_CHAR_SETS = dict()


# EN
en_valid_char_set = set(list(
    string.ascii_letters +
    string.digits +
    string.punctuation
) + [chr(i) for i in range(0x20A0, 0x20CF + 1)])  # Currency symbols
VALID_CHAR_SETS['en'] = en_valid_char_set


# DE
de_char_range = [196, 228, 214, 246, 220, 252, 7838, 223]
de_char_set = set(map(chr, de_char_range))
de_valid_char_set = de_char_set | en_valid_char_set
VALID_CHAR_SETS['de'] = de_valid_char_set


# ZH
zh_char_range = itertools.chain(
    range(0x4E00, 0x9FFF + 1),  # CJK Unified Ideographs
    range(0x3400, 0x4DBF + 1),  # CJK Unified Ideographs Extension A
    range(0x20000, 0x2A6DF + 1),  # CJK Unified Ideographs Extension B
    range(0x2A700, 0x2B73F + 1),  # CJK Unified Ideographs Extension C
    range(0x2B740, 0x2B81F + 1),  # CJK Unified Ideographs Extension D
    range(0x2B820, 0x2CEAF + 1),  # CJK Unified Ideographs Extension E
    range(0x2CEB0, 0x2EBEF + 1),  # CJK Unified Ideographs Extension F
    range(0xF900, 0xFAFF + 1),  # CJK Compatibility Ideographs
    range(0x2F800, 0x2FA1F + 1),  # CJK Compatibility Ideographs Supplement
    range(0x9FA6, 0x9FCB + 1)  # Small Extensions to the URO
)
zh_char_set = set(map(chr, zh_char_range))
zh_valid_char_set = zh_char_set | en_valid_char_set
VALID_CHAR_SETS['zh'] = zh_valid_char_set


# TH
th_char_range = range(0x0E00, 0x0E7F + 1)
th_char_set = set(map(chr, th_char_range))
th_valid_char_set = th_char_set | en_valid_char_set
VALID_CHAR_SETS['th'] = th_valid_char_set


# VN
vn_char_range = itertools.chain(
    range(0x0041, 0x005A + 1),
    range(0x0061, 0x007A + 1),
    range(0x00C0, 0x00C3 + 1),
    range(0x00C8, 0x00CA + 1),
    range(0x00CC, 0x00CD + 1),
    range(0x00D2, 0x00D5 + 1),
    range(0x00D9, 0x00DA + 1),
    range(0x00DD, 0x00DD + 1),
    range(0x00E0, 0x00E3 + 1),
    range(0x00E8, 0x00EA + 1),
    range(0x00EC, 0x00ED + 1),
    range(0x00F2, 0x00F5 + 1),
    range(0x00F9, 0x00FA + 1),
    range(0x00FD, 0x00FD + 1),
    range(0x0102, 0x0103 + 1),
    range(0x0110, 0x0111 + 1),
    range(0x0128, 0x0129 + 1),
    range(0x0168, 0x0169 + 1),
    range(0x01A0, 0x01A1 + 1),
    range(0x01AF, 0x01B0 + 1),
    range(0x1EA0, 0x1EF9 + 1),
)
vn_char_set = set(map(chr, vn_char_range))
vn_valid_char_set = vn_char_set | en_valid_char_set
VALID_CHAR_SETS['vn'] = vn_valid_char_set
