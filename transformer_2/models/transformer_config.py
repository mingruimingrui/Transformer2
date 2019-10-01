"""
Configurables for Transformer
"""

from transformers_2.utils.config_system import ConfigSystem

__all__ = ['make_config']


def validate_config(config):
    pass


_C = ConfigSystem(validate_config_fn=validate_config)
# --------------------------------------------------------------------------- #
# Start of configs
# --------------------------------------------------------------------------- #

# Required
# The vocab size of the encoder language
# Should be the number of tokens including <bos> <pad> <eos> <unk>
_C.encoder_vocab_size = None

# Required
# The index of the padding token
_C.encoder_padding_idx = None

# The size of the token/positional embeddings for the encoder
_C.encoder_embed_dim = 512

# The size of the hidden states embeddings in the encoder
_C.encoder_hidden_dim = 512

# The size of the hidden states in the encoder transistor
_C.encoder_transistor_dim = 2048

# The number of multi-head attention layers
_C.encoder_num_layers = 6

# The number of heads in multi-head attention
_C.encoder_num_heads = 8

# Should bias be used in the encoder
_C.encoder_use_bias = True

# The number of positional embeddings to use
_C.encoder_max_positions = 1024

# Should positional embeddings not be used?
_C.encoder_no_pos_embeds = False

# Should positional embeddings be learned? Default uses sinusoidal
_C.encoder_learned_pos_embeds = False

# Required
# The vocab size of the decoder language
# Should be the number of tokens including <bos> <pad> <eos> <unk>
_C.decoder_vocab_size = None

# Required
# The index of the padding token
_C.decoder_padding_idx = None

# The size of the token/positional embeddings for the encoder
_C.decoder_embed_dim = 512

# The size of the hidden states embeddings in the decoder
_C.encoder_hidden_dim = 512

# The size of the hidden states in the decoder transistor
_C.decoder_transistor_dim = 2048

# The number of multi-head attention layers
_C.decoder_num_layers = 6

# The number of heads in multi-head attention
_C.decoder_num_heads = 8

# Should bias be used in the decoder
_C.decoder_use_bias = True

# The number of positional embeddings to use
_C.decoder_max_positions = 1024

# Should positional embeddings not be used?
_C.decoder_no_pos_embeds = False

# Should positional embeddings be learned? Default uses sinusoidal
_C.decoder_learned_pos_embeds = False

# Should the decoder not attend to the encoder? Default the
# decoder will attend to the encoder.
_C.decoder_no_encoder_attn = False

# Dropout probability
_C.dropout = 0.0

# Dropout probability for attention weights
_C.attn_dropout = 0.0

# Dropout probability after attention in transistor
_C.activation_dropout = 0.0

# Activation function to use in transistor
_C.activation_fn = 'relu'

# Should layer norm be applied before multi-headed attention?
# Default is after
_C.normalize_before = False

# Should encoder input embeddings, decoder input embeddings and decoder output
# embeddings be the same tensor?
_C.share_all_embeddings = False

# Should decoder input and output embeddings be the same tensor?
_C.share_decoder_input_output_embed = True

# --------------------------------------------------------------------------- #
# End of configs
# --------------------------------------------------------------------------- #
_C.immutable(True)
make_config = _C.make_config
