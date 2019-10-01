"""
Script that contains sentencepiece tokenizer as processors
"""

from __future__ import absolute_import
from __future__ import unicode_literals

from ._registry import BaseProcessor, register_processor

__all__ = ['SpmEncode', 'SpmDecode']


@register_processor('spm_encode')
class SpmEncode(BaseProcessor):
    """
    Processor that performs sentencepiece encoding with a model
    Args:
        spm_model_path: path to a sentencepiece model
        form: Choice of 'pieces' and 'ids'. This processor would encode
            text into either pieces or ids
    """

    def __init__(self, spm_model_path, form='pieces'):
        try:
            import sentencepiece
        except ImportError:
            raise ImportError('Please install package `sentencepiece`')
        self.spm_model = sentencepiece.SentencePieceProcessor()
        self.spm_model.Load(spm_model_path)
        if form == 'pieces':
            self.form_is_pieces = True
        elif form == 'ids':
            self.form_is_pieces = False
        else:
            raise ValueError('form unrecognized, {}'.format(form))

    def __call__(self, text):
        if self.form_is_pieces:
            return ' '.join(self.spm_model.EncodeAsPieces(text))
        else:
            return self.spm_model.EncodeAsIds(text)


@register_processor('spm_decode')
class SpmDecode(BaseProcessor):
    """
    Processor that performs sentencepiece decoding with a model
    Args:
        spm_model_path: path to a sentencepiece model
        form: Choice of 'pieces' and 'ids'. This processor would encode
            text into either pieces or ids
    """

    def __init__(self, spm_model_path, form='pieces'):
        try:
            import sentencepiece
        except ImportError:
            raise ImportError('Please install package `sentencepiece`')
        self.spm_model = sentencepiece.SentencePieceProcessor()
        self.spm_model.Load(spm_model_path)
        if form == 'pieces':
            self.form_is_pieces = True
        elif form == 'ids':
            self.form_is_pieces = False
        else:
            raise ValueError('form unrecognized, {}'.format(form))

    def __call__(self, text_or_list_of_ids):
        if self.form_is_pieces:
            return self.spm_model.DecodePieces(text_or_list_of_ids.split())
        else:
            return self.spm_model.DecodeIds(text_or_list_of_ids)
