import math
import numpy as np
import tensorflow as tf

from transformer_2.layers.multihead_attention \
    import CachedMultiheadAttention, IncrementalMultiheadAttention
from transformer_2.layers.transistor import Transistor
from transformer_2.utils.sinusoidal_positional_embedding \
    import build_sinusoidal_positional_embedding


class TransformerDecoderLayer(tf.keras.Model):
    """
    Transformer decoder layer block
    """

    def __init__(
        self,
        hidden_dim: int = 512, transistor_dim: int = 2048,
        num_heads: int = 6, use_bias: bool = True,
        dropout: float = 0.0, attn_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_fn='relu', normalize_before: bool = False,
        name: str = 'decoder_layer', **kwargs
    ):
        super(TransformerDecoderLayer, self).__init__(name=name, **kwargs)

        # Constants
        self.dropout = dropout
        self.normalize_before = normalize_before

        # Layers
        self.layer_norm = tf.keras.layers.LayerNormalization(**kwargs)
        self.attn_layer = IncrementalMultiheadAttention(
            hidden_dim=hidden_dim, num_heads=num_heads,
            use_bias=use_bias, attn_dropout=attn_dropout,
            trainable=True, **kwargs
        )
        self.encoder_attn_layer = CachedMultiheadAttention(
            hidden_dim=hidden_dim, num_heads=num_heads,
            use_bias=use_bias, attn_dropout=attn_dropout,
            trainable=True, **kwargs
        )
        self.transistor = Transistor(
            inner_dim=transistor_dim, activation_fn=activation_fn,
            use_bias=use_bias, activation_dropout=activation_dropout,
            trainable=True, **kwargs
        )

    def call(self, inputs, training=None):
        (
            x, attn_mask,
            encoder_out, encoder_padding_mask,
            incremental_state, new_state_order
        ) = inputs

        # Apply self attention
        residual = x
        x = self.layer_norm(x) if self.normalize_before else x

        x, incremental_state = self.attn_layer((
            x, x, x, attn_mask,
            incremental_state, new_state_order
        ), training=training)

        x = tf.nn.dropout(x, rate=self.dropout) if training else x
        x += residual
        x = x if self.normalize_before else self.layer_norm(x)

        # Apply attn on encoder out
        residual = x
        x = self.layer_norm(x) if self.normalize_before else x

        x, incremental_state = self.encoder_attn_layer((
            x, encoder_out, encoder_out, encoder_padding_mask,
            incremental_state, new_state_order
        ), training=training)

        x = tf.nn.dropout(x, rate=self.dropout) if training else x
        x += residual
        x = x if self.normalize_before else self.layer_norm(x)

        # Apply fc1 and fc2
        residual = x
        x = self.layer_norm(x) if self.normalize_before else x

        x = self.transistor(x, training=training)

        x = tf.nn.dropout(x, rate=self.dropout) if training else x
        x += residual
        x = x if self.normalize_before else self.layer_norm(x)

        return x, incremental_state


class TransformerDecoder(tf.keras.Model):
    """
    Custom implementation of transformer decoder
    designed with incremental decoding in mind
    """

    def __init__(
        self,
        in_token_embeddings: tf.keras.layers.Embedding,
        out_token_embeddings: tf.keras.layers.Embedding, padding_idx: int,
        no_pos_embeds: bool = False, max_positions: int = 1024,
        learned_pos_embeds: bool = False, num_layers: int = 6,
        hidden_dim: int = 512, transistor_dim: int = 2048,
        num_heads: int = 6, use_bias: bool = True,
        dropout: float = 0.0, attn_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_fn='relu', normalize_before: bool = False,
        name: str = 'transformer_decoder', **kwargs
    ):
        super(TransformerDecoder, self).__init__(name=name, **kwargs)
        assert isinstance(in_token_embeddings, tf.keras.layers.Embedding)
        input_dim = in_token_embeddings.output_dim
        assert isinstance(out_token_embeddings, tf.keras.layers.Embedding)
        output_dim = out_token_embeddings.output_dim

        self.share_input_output_embeddings = \
            in_token_embeddings == out_token_embeddings

        # Initialize out token embeddings at start
        out_token_embeddings(tf.constant([[0]]))

        # Constant
        self.no_pos_embeds = no_pos_embeds
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.embed_scale = math.sqrt(input_dim)
        self.normalize_before = normalize_before

        def future_mask_initializer(shape, dtype):
            m = np.ones([int(i) for i in shape], dtype=dtype.name)
            return np.triu(m, k=1)
        self.future_mask = self.add_weight(
            name='future_mask',
            shape=[max_positions, max_positions],
            dtype=self.dtype,
            initializer=future_mask_initializer,
            trainable=False
        )

        # Layers
        if self.share_input_output_embeddings:
            self.token_embeddings = in_token_embeddings
        else:
            self.in_token_embeddings = in_token_embeddings
            self.out_token_embeddings = out_token_embeddings
        if not no_pos_embeds:
            def positional_embeddings_initializer(shape, dtype):
                num_embeddings = int(shape[0])
                embedding_dim = int(shape[1])
                return build_sinusoidal_positional_embedding(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                    dtype=dtype.name
                )
            self.position_embeddings = tf.keras.layers.Embedding(
                input_dim=max_positions,
                output_dim=input_dim,
                mask_zero=False,
                embeddings_initializer=positional_embeddings_initializer,
                trainable=learned_pos_embeds,
                **kwargs
            )
            # Initialize layer here to generate positional embedding weights
            self.position_embeddings(tf.constant([[0]]))
        self.project_out_dim = None if hidden_dim == output_dim else \
            tf.keras.layers.Dense(output_dim, use_bias=False)
        self.decoder_layers = []
        for i in range(num_layers):
            self.decoder_layers.append(TransformerDecoderLayer(
                hidden_dim=hidden_dim, transistor_dim=transistor_dim,
                num_heads=num_heads, use_bias=use_bias,
                dropout=dropout, attn_dropout=attn_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn, normalize_before=normalize_before,
                name='decoder_layer_{}'.format(i), **kwargs
            ))
        if normalize_before:
            self.layer_norm = tf.keras.layers.LayerNormalization(**kwargs)

    def call(self, inputs, training=None):
        (
            prev_output_tokens,
            encoder_out, encoder_padding_mask,
            incremental_state, new_state_order
        ) = inputs

        src_len = tf.shape(prev_output_tokens)[1]

        # Get x
        if self.share_input_output_embeddings:
            x = self.token_embeddings(prev_output_tokens)
        else:
            x = self.in_token_embeddings(prev_output_tokens)
        x *= self.embed_scale
        if not self.no_pos_embeds:
            x += self.position_embeddings.embeddings[:src_len]
        x = tf.nn.dropout(x, self.dropout) if training else x

        # B x T x C -> T x B x C
        x = tf.transpose(x, [1, 0, 2])

        # Apply decoder layers
        for layer in self.decoder_layers:
            x, incremental_state = layer((
                x, self.future_mask[:src_len, :src_len],
                encoder_out, encoder_padding_mask,
                incremental_state, new_state_order
            ), training=training)

        if self.normalize_before:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = tf.transpose(x, [1, 0, 2])

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        x = tf.matmul(
            x, self.out_token_embeddings.embeddings,
            transpose_b=True
        )

        return x, incremental_state
