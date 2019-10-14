import math
import tensorflow as tf

from transformer_2.layers.multihead_attention import MultiheadAttention
from transformer_2.layers.transistor import Transistor
from transformer_2.utils.sinusoidal_positional_embedding \
    import build_sinusoidal_positional_embedding


class TransformerEncoderLayer(tf.keras.Model):
    """
    Transformer encoder layer block
    """

    def __init__(
        self,
        hidden_dim: int = 512, transistor_dim: int = 2048,
        num_heads: int = 6, use_bias: bool = True,
        dropout: float = 0.0, attn_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_fn='relu', normalize_before: bool = False,
        name: str = 'encoder_layer', **kwargs
    ):
        super(TransformerEncoderLayer, self).__init__(name=name, **kwargs)

        # Constants
        self.dropout = dropout
        self.normalize_before = normalize_before

        # Layers
        self.attn_layer = MultiheadAttention(
            hidden_dim=hidden_dim, num_heads=num_heads,
            use_bias=use_bias, attn_dropout=attn_dropout,
            trainable=True, name='self_attn', **kwargs
        )
        self.attn_layer_norm = tf.keras.layers.LayerNormalization(
            name='self_attn_layer_norm', **kwargs
        )
        self.transistor = Transistor(
            inner_dim=transistor_dim, activation=activation_fn,
            use_bias=use_bias, activation_dropout=activation_dropout,
            trainable=True, name='transistor', **kwargs
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            name='layer_norm', **kwargs
        )

    def call(self, inputs, training=None):
        x, key_padding_mask = inputs

        # Apply multihead attn
        residual = x
        x = self.attn_layer_norm(x) if self.normalize_before else x

        x = self.attn_layer((x, x, x, key_padding_mask), training=training)

        x = tf.nn.dropout(x, rate=self.dropout) if training else x
        x += residual
        x = x if self.normalize_before else self.attn_layer_norm(x)

        # Apply transistor
        residual = x
        x = self.layer_norm(x) if self.normalize_before else x

        x = self.transistor(x, training=training)

        x = tf.nn.dropout(x, rate=self.dropout) if training else x
        x += residual
        x = x if self.normalize_before else self.layer_norm(x)

        return x


class TransformerEncoder(tf.keras.Model):
    """
    Transformer encoder
    """

    def __init__(
        self,
        token_embeddings: tf.keras.layers.Embedding, padding_idx: int,
        no_pos_embeds: bool = False, max_positions: int = 1024,
        learned_pos_embeds: bool = False, num_layers: int = 6,
        hidden_dim: int = 512, transistor_dim: int = 2048,
        num_heads: int = 6, use_bias: bool = True,
        dropout: float = 0.0, attn_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_fn='relu', normalize_before: bool = False,
        name: str = 'transformer_encoder', **kwargs
    ):
        super(TransformerEncoder, self).__init__(name=name, **kwargs)
        assert isinstance(token_embeddings, tf.keras.layers.Embedding)
        embed_dim = token_embeddings.output_dim

        # Constants
        self.no_pos_embeds = no_pos_embeds
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.embed_scale = math.sqrt(embed_dim)
        self.normalize_before = normalize_before

        # Layers
        self.token_embeddings = token_embeddings
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
                input_dim=max_positions, output_dim=embed_dim, mask_zero=False,
                embeddings_initializer=positional_embeddings_initializer,
                trainable=learned_pos_embeds, name='pos_embeds', **kwargs
            )
            # Initialize layer here to generate positional embedding weights
            self.position_embeddings(tf.constant([[0]]))

        self.project_in_dim = None if hidden_dim == embed_dim else \
            tf.keras.layers.Dense(hidden_dim, use_bias=False, **kwargs)

        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(TransformerEncoderLayer(
                hidden_dim=hidden_dim, transistor_dim=transistor_dim,
                num_heads=num_heads, use_bias=use_bias,
                dropout=dropout, attn_dropout=attn_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn, normalize_before=normalize_before,
                name='layer_{}'.format(i), **kwargs
            ))
        if normalize_before:
            self.layer_norm = tf.keras.layers.LayerNormalization(
                name='layer_norm', **kwargs
            )

    def call(self, inputs, training=None):
        src_tokens, src_lengths = inputs

        # Get x
        x = self.embed_scale * self.token_embeddings(src_tokens)
        if not self.no_pos_embeds:
            x += self.position_embeddings.embeddings[:tf.shape(x)[1]]
        x = tf.nn.dropout(x, self.dropout) if training else x

        # Re-rank to hidden dim size
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        # B x T x C -> T x B x C
        x = tf.transpose(x, [1, 0, 2])

        # Compute padding mask
        padding_mask = tf.equal(src_tokens, self.padding_idx)

        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer((x, padding_mask), training=training)

        if self.normalize_before:
            x = self.layer_norm(x)

        return x, padding_mask
