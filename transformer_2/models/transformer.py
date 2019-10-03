import warnings
import tensorflow as tf

from transformer_2.models.transformer_config import make_config
from transformer_2.models.transformer_encoder import TransformerEncoder
from transformer_2.models.transformer_decoder import TransformerDecoder


class Transformer(tf.keras.Model):
    def __init__(self, encoder, decoder, config, name='transformer', **kwargs):
        super(Transformer, self).__init__(name=name, **kwargs)
        assert isinstance(encoder, TransformerEncoder)
        assert isinstance(decoder, TransformerDecoder)
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def call(self, inputs, training=None):
        (
            src_tokens, src_lengths, prev_output_tokens,
            incremental_state, new_state_order
        ) = inputs

        encoder_out, encoder_padding_mask = self.encoder((
            src_tokens, src_lengths
        ))
        decoder_out, incremental_state = self.decoder((
            prev_output_tokens,
            encoder_out, encoder_padding_mask,
            incremental_state, new_state_order
        ))

        return decoder_out, incremental_state

    @classmethod
    def make_model(
        cls,
        config_file=None, config_dict=None,
        name='transformer', **kwargs
    ):
        assert config_file is not None or config_dict is not None
        config = make_config(config_file=config_file, **config_dict)

        # Make encoder embeddings
        encoder_token_embeddings = tf.keras.layers.Embedding(
            input_dim=config.encoder_vocab_size,
            output_dim=config.encoder_embed_dim,
            mask_zero=False, name='token_embeds', **kwargs)

        # Make decoder input and output embeddings
        if config.share_all_embeddings:
            if not config.share_decoder_input_output_embed:
                warnings.warn(
                    'As share_all_embeddings is specified, '
                    'share_decoder_input_output_embed is assumed.'
                )
            decoder_in_token_embeddings = encoder_token_embeddings
            decoder_out_token_embeddings = encoder_token_embeddings

        else:
            decoder_in_token_embeddings = tf.keras.layers.Embedding(
                input_dim=config.decoder_vocab_size,
                output_dim=config.decoder_embed_dim,
                mask_zero=False, name='token_embeds', **kwargs)

            if config.share_decoder_input_output_embed:
                decoder_out_token_embeddings = decoder_in_token_embeddings
            else:
                decoder_out_token_embeddings = tf.keras.layers.Embedding(
                    input_dim=config.decoder_vocab_size,
                    output_dim=config.decoder_embed_dim,
                    mask_zero=False, name='out_token_embeds', **kwargs)

        encoder = TransformerEncoder(
            token_embeddings=encoder_token_embeddings,
            padding_idx=config.encoder_padding_idx,
            no_pos_embeds=config.encoder_no_pos_embeds,
            max_positions=config.encoder_max_positions,
            learned_pos_embeds=config.encoder_learned_pos_embeds,
            num_layers=config.encoder_num_layers,
            hidden_dim=config.encoder_hidden_dim,
            transistor_dim=config.encoder_transistor_dim,
            num_heads=config.encoder_num_heads,
            use_bias=config.encoder_use_bias,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            activation_dropout=config.activation_dropout,
            activation_fn=config.activation_fn,
            normalize_before=config.normalize_before,
            **kwargs
        )

        decoder = TransformerDecoder(
            in_token_embeddings=decoder_in_token_embeddings,
            out_token_embeddings=decoder_out_token_embeddings,
            padding_idx=config.decoder_padding_idx,
            no_pos_embeds=config.decoder_no_pos_embeds,
            max_positions=config.decoder_max_positions,
            learned_pos_embeds=config.decoder_learned_pos_embeds,
            num_layers=config.decoder_num_layers,
            hidden_dim=config.decoder_hidden_dim,
            transistor_dim=config.decoder_transistor_dim,
            num_heads=config.decoder_num_heads,
            use_bias=config.decoder_use_bias,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            activation_dropout=config.activation_dropout,
            activation_fn=config.activation_fn,
            normalize_before=config.normalize_before,
            **kwargs
        )

        return Transformer(encoder, decoder, config, name=name, **kwargs)
