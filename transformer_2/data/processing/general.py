"""
Script that contains some common everyday processors
"""

from __future__ import absolute_import
from __future__ import unicode_literals

import html
import unicodedata

from transformer_2.data.processing._registry \
    import BaseProcessor, register_processor

__all__ = [
    'HtmlUnescape', 'HtmlEscape', 'Lowercase', 'Strip',
    'WhitespaceNormalize', 'UnicodeNormalize'
]


@register_processor('html_unescape')
class HtmlUnescape(BaseProcessor):
    """
    Processor that performs html unescape on text
    Eg. &lt to <;
        &gt to >;
        &amp to &;
        And more
    """

    def __init__(self):
        pass

    def __call__(self, text):
        return html.unescape(text)


@register_processor('html_escape')
class HtmlEscape(BaseProcessor):
    """
    Processor that performs html escape on text
    Eg. < to &lt;
        > to &gt;
        & to &amp;
        And more
    """

    def __init__(self):
        pass

    def __call__(self, text):
        return html.escape(text)


@register_processor('lowercase')
class Lowercase(BaseProcessor):
    """ Casts alphabets to lowercase """

    def __init__(self):
        pass

    def __call__(self, text):
        return text.lower()


@register_processor('strip')
class Strip(BaseProcessor):
    """ Removes leading and trailing whitespaces """

    def __init__(self):
        pass

    def __call__(self, text):
        return text.strip()


@register_processor('whitespace_normalize')
class WhitespaceNormalize(BaseProcessor):
    """ Removes repeated whitespace """

    def __init__(self):
        pass

    def __call__(self, text):
        return ' '.join(text.split())


@register_processor('unicode_normalize')
class UnicodeNormalize(BaseProcessor):
    """
    Normalize characters into their base forms based on the unicode
    based on the unicode standard https://unicode.org/reports/tr15/
    """

    def __init__(self, form='NFKC'):
        self.form = form

    def __call__(self, text):
        return unicodedata.normalize(self.form, text)


@register_processor('unidecode')
class Unidecode(BaseProcessor):
    """
    Perform character normalization using the unidecode package
    """

    def __init__(self):
        pass

    def __call__(self, text):
        try:
            from unidecode import unidecode_expect_nonascii
        except ImportError:
            raise ImportError('Please install package `unidecode`')
        return unidecode_expect_nonascii(text)
