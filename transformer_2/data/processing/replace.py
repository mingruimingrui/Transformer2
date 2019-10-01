"""
Script that contains processors to do string replacements
"""

from __future__ import absolute_import
from __future__ import unicode_literals

import os
import re
from typing import Mapping, Iterable
from six import string_types, text_type

from transformer_2.data.processing._registry \
    import BaseProcessor, register_processor
from transformer_2.utils.io_utils import read_json_from_file
from transformer_2.utils.char_sets import VALID_CHAR_SETS

__all__ = ['ReplaceSubstrings', 'ReplaceTokens', 'WhitelistCharacters']


def _get_replace_dict(replace_dict_or_filepath):
    if isinstance(replace_dict_or_filepath, string_types):
        # arg is a filepath
        # Extract actual replace_dict from file
        filepath = replace_dict_or_filepath
        replace_dict = read_json_from_file(filepath)

    elif isinstance(replace_dict, Mapping):
        replace_dict = replace_dict_or_filepath

    else:
        raise ValueError(
            'replace_dict of unrecognized format, '
            'expecting either a filepath or a dictionary'
        )

    # Cast replace dict to unicode
    def cast_to_unicode(s):
        """ Cast python string to unicode string """
        if not isinstance(s, text_type):
            s = s.decode('utf-8')
        return s

    _replace_dict = dict()
    for k, v in replace_dict.items():
        err_msg = 'Expecting a unicode string but got {} ({})'
        assert isinstance(k, string_types), err_msg.format(k, type(k))
        assert isinstance(v, string_types), err_msg.format(v, type(v))
        _replace_dict[cast_to_unicode(k)] = cast_to_unicode(v)
    return _replace_dict


@register_processor('replace_substrings')
class ReplaceSubstrings(BaseProcessor):
    """
    Processor that performs substring replacement using some dictionary
    Args:
        replace_dict: Can be 1 of the following 3 types
            - The path to a json file containing a dictionary object
            - A python dictionary/Mapping object
    """

    def __init__(self, replace_dict):
        # Get replace_dict
        replace_dict = _get_replace_dict(replace_dict)

        pattern_string = '|'.join(re.escape(k) for k in replace_dict.keys())
        pattern = re.compile(pattern_string)

        self.replace_dict = replace_dict
        self.pattern = pattern

    def _replace_fn(self, match):
        return self.replace_dict(match.group(0))

    def __call__(self, text):
        return self.pattern.sub(self._replace_fn, text)


@register_processor('replace_tokens')
class ReplaceTokens(BaseProcessor):
    """
    Processor that performs token replacement using some dictionary
    Args:
        replace_dict: Can be 1 of the following 3 types
            - The path to a json file containing a dictionary object
            - A python dictionary/Mapping object

    Note check the default language char_sets in
    transformer_2/utils/char_sets.py
    """

    def __init__(self, replace_dict):
        self.replace_dict = _get_replace_dict(replace_dict)

    def __call__(self, text):
        return ' '.join(self.replace_dict.get(t, t) for t in text.split())


@register_processor('whitelist_characters')
class WhitelistCharacters(BaseProcessor):
    """
    Processor that replaces non-whitelisted characters with a replacement
    string (whitespace by default)
    Args:
        char_set: Can be 1 of the following 3 types
            - One of {}
            - The path to a json file containing a list of valid characters
            - A list/set/Iterable of characters
        replace_string: The string to replace non-whitelisted characters
    """.format(list(VALID_CHAR_SETS.keys()))

    def __init__(self, char_set, replace_string=' '):
        # Cast char_set to an iterable set of characters
        if isinstance(char_set, string_types):
            if char_set in VALID_CHAR_SETS:
                char_set = VALID_CHAR_SETS[char_set]
            elif os.path.isfile(char_set):
                filepath = char_set
                char_set = read_json_from_file(filepath)

        assert isinstance(char_set, Iterable), (
            'Expecting char_set to be an iterable, instead received {}'
        ).format(type(char_set))

        char_set = set(c for c in char_set)
        for c in char_set:
            assert isinstance(c, text_type), \
                'Expecting unicode string but received {}'.format(type(c))

        assert isinstance(replace_string, text_type), (
            'Exepcting replace_string to be a unicode string, instead got {}'
        ).format(type(replace_string))

        self.char_set = char_set
        self.replace_string = replace_string

    def __call__(self, text):
        return ''.join(
            c if c in self.char_set else self.replace_string
            for c in text
        )
