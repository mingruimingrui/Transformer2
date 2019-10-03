import tensorflow as tf
from transformer_2.utils import incr_decoding_utils


class MultiheadAttention(tf.keras.layers.Layer):
    """
    For self-attention in encoder

    inputs
        - query_input (tgt_len, bsz, hidden_dim)
        - key_input (src_len, bsz, hidden_dim)
        - value_input (src_len, bsz, hidden_dim)
        - key_padding_mask (bsz, src_len) : 1 at positions to mask, 0 at others
    outputs
        - attn_output (tgt_len, bsz, hidden_dim)
        - attn_output_weights (bsz, tgt_len, src_len) : requires need_weights
    """

    def __init__(
        self,
        hidden_dim: int, num_heads: int,
        use_bias: bool = True, attn_dropout: float = 0.0,
        **kwargs
    ):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.attn_dropout = attn_dropout

        self.head_dim = self.hidden_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        assert self.head_dim * self.num_heads == self.hidden_dim, \
            'hidden_dim must be divisible by num_heads'

    def build(self, input_shapes):
        query_input_shape = input_shapes[0]
        input_dim = int(query_input_shape[2])

        # Input and output linear layer weights
        self.in_q_proj_weight = self.add_weight(
            name='in_q_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.in_k_proj_weight = self.add_weight(
            name='in_k_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.in_v_proj_weight = self.add_weight(
            name='in_v_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )

        self.out_proj_weight = self.add_weight(
            name='out_proj_weight',
            shape=[self.hidden_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )

        # Input and output linear layer bias
        if self.use_bias:
            self.in_q_proj_bias = self.add_weight(
                name='in_q_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )
            self.in_k_proj_bias = self.add_weight(
                name='in_k_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )
            self.in_v_proj_bias = self.add_weight(
                name='in_v_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )

            self.out_proj_bias = self.add_weight(
                name='out_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )

        # A negative infinity will be multiplied to masks and added to
        # tensors so that after softmax, the areas masked will have value of 0
        # As inf is unstable in tf. ie. 0 * inf = nan
        # The max value of the model dtype will be used
        if isinstance(self.dtype, tf.DType):
            self.INF = self.dtype.max
        else:
            # isinstance(self.dtype, str):
            dtype = getattr(tf, self.dtype)
            self.INF = dtype.max

        super(MultiheadAttention, self).build(input_shapes)

    def call(self, inputs, training=None, need_weights=None):
        query_input, key_input, value_input, key_padding_mask = inputs

        query_input_shape = tf.shape(query_input)
        tgt_len = query_input_shape[0]
        bsz = query_input_shape[1]

        # Compute query, key and value tensors
        q = tf.matmul(query_input, self.in_q_proj_weight, transpose_b=True)
        k = tf.matmul(key_input, self.in_k_proj_weight, transpose_b=True)
        v = tf.matmul(value_input, self.in_v_proj_weight, transpose_b=True)
        if self.use_bias:
            q += self.in_q_proj_bias
            k += self.in_k_proj_bias
            v += self.in_v_proj_bias
        q *= self.scaling

        # Reshape into (batch_size * num_heads, seq_len, head_dim)
        q = tf.reshape(q, [-1, bsz * self.num_heads, self.head_dim])
        q = tf.transpose(q, [1, 0, 2])
        k = tf.reshape(k, [-1, bsz * self.num_heads, self.head_dim])
        k = tf.transpose(k, [1, 0, 2])
        v = tf.reshape(v, [-1, bsz * self.num_heads, self.head_dim])
        v = tf.transpose(v, [1, 0, 2])

        src_len = tf.shape(k)[1]

        # Compute attention weights also mask attn if needed
        attn_output_weights = tf.linalg.matmul(q, tf.transpose(k, [0, 2, 1]))
        # attn_output_weights_shape = [bsz * num_heads, tgt_len, src_len]

        # Don't attend to padding symbols
        key_padding_mask = \
            tf.expand_dims(tf.expand_dims(key_padding_mask, 1), 2)
        attn_output_weights = tf.reshape(
            attn_output_weights, [bsz, self.num_heads, tgt_len, src_len])
        attn_output_weights = attn_output_weights - self.INF * \
            tf.cast(key_padding_mask, attn_output_weights.dtype)
        attn_output_weights = tf.reshape(
            attn_output_weights, [bsz * self.num_heads, tgt_len, src_len])

        # Apply softmax to get attention distribution
        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=2)
        if training:
            attn_output_weights = \
                tf.nn.dropout(attn_output_weights, rate=self.attn_dropout)

        # Apply attention weights to value to get outputs
        attn_output = tf.matmul(attn_output_weights, v)

        # Reshape back into (seq_len, batch_size, hidden_dim)
        attn_output = tf.transpose(attn_output, [1, 0, 2])
        attn_output = tf.reshape(attn_output, [tgt_len, bsz, self.hidden_dim])

        # Apply final linear layer
        attn_output = tf.matmul(
            attn_output, self.out_proj_weight,
            transpose_b=True
        )
        if self.use_bias:
            attn_output += self.out_proj_bias

        if need_weights:
            attn_output_weights = tf.reshape(
                attn_output_weights, [bsz, self.num_heads, tgt_len, src_len])
            attn_output_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, attn_output_weights
        else:
            return attn_output


class CachedMultiheadAttention(tf.keras.layers.Layer):
    """
    For alignment in decoder

    inputs
        - query_input (tgt_len, bsz, hidden_dim)
        - key_input (src_len, bsz, hidden_dim)
        - value_input (src_len, bsz, hidden_dim)
        - key_padding_mask (bsz, src_len) : 1 at positions to mask, 0 at others
        - incremental_state (dict)
        - new_state_order (bsz)
    outputs
        - attn_output (tgt_len, bsz, hidden_dim)
        - incremental_state (dict)
        - attn_output_weights (bsz, tgt_len, src_len) : requires need_weights
    """

    _key_postfix = 'cached_state'

    def __init__(
        self,
        hidden_dim: int, num_heads: int,
        use_bias: bool = True, attn_dropout: float = 0.0,
        **kwargs
    ):
        super(CachedMultiheadAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.attn_dropout = attn_dropout

        self.head_dim = self.hidden_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        assert self.head_dim * self.num_heads == self.hidden_dim, \
            'hidden_dim must be divisible by num_heads'

    def build(self, input_shapes):
        query_input_shape = input_shapes[0]
        input_dim = int(query_input_shape[2])

        # Input and output linear layer weights
        self.in_q_proj_weight = self.add_weight(
            name='in_q_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.in_k_proj_weight = self.add_weight(
            name='in_k_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.in_v_proj_weight = self.add_weight(
            name='in_v_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )

        self.out_proj_weight = self.add_weight(
            name='out_proj_weight',
            shape=[self.hidden_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )

        # Input and output linear layer bias
        if self.use_bias:
            self.in_q_proj_bias = self.add_weight(
                name='in_q_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )
            self.in_k_proj_bias = self.add_weight(
                name='in_k_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )
            self.in_v_proj_bias = self.add_weight(
                name='in_v_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )

            self.out_proj_bias = self.add_weight(
                name='out_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )

        # A negative infinity will be multiplied to masks and added to
        # tensors so that after softmax, the areas masked will have value of 0
        # As inf is unstable in tf. ie. 0 * inf = nan
        # The max value of the model dtype will be used
        if isinstance(self.dtype, tf.DType):
            self.INF = self.dtype.max
        else:
            # isinstance(self.dtype, str):
            dtype = getattr(tf, self.dtype)
            self.INF = dtype.max

        super(CachedMultiheadAttention, self).build(input_shapes)

    def _get_saved_state(self, incremental_state):
        return incr_decoding_utils.get_state(
            module_instance=self,
            incremental_state=incremental_state,
            key_postfix=self._key_postfix
        ) or {}

    def _set_saved_state(self, incremental_state, value):
        incr_decoding_utils.set_state(
            module_instance=self,
            incremental_state=incremental_state,
            key_postfix=self._key_postfix,
            value=value
        )

    def call(self, inputs, training=None, need_weights=None):
        (
            query_input, key_input, value_input, key_padding_mask,
            incremental_state, new_state_order
        ) = inputs

        query_input_shape = tf.shape(query_input)
        tgt_len = query_input_shape[0]
        bsz = query_input_shape[1]

        # Get previous state
        saved_state = self._get_saved_state(incremental_state)

        # Compute query tensor
        q = tf.matmul(query_input, self.in_q_proj_weight, transpose_b=True)
        if self.use_bias:
            q += self.in_q_proj_bias
        q *= self.scaling

        # Reshape into (batch_size * num_heads, seq_len, head_dim)
        q = tf.reshape(q, [-1, bsz * self.num_heads, self.head_dim])
        q = tf.transpose(q, [1, 0, 2])

        if 'prev_key' in saved_state:
            # Gather key and value tensor from saved_state
            k = saved_state['prev_key']
            v = saved_state['prev_value']

            k = tf.gather(k, new_state_order)
            v = tf.gather(v, new_state_order)

            # Reshape into (batch_size * num_heads, seq_len, head_dim)
            k = tf.reshape(k, [bsz * self.num_heads, -1, self.head_dim])
            v = tf.reshape(v, [bsz * self.num_heads, -1, self.head_dim])
        else:
            # Compute key and value tensors
            k = tf.matmul(key_input, self.in_k_proj_weight, transpose_b=True)
            v = tf.matmul(value_input, self.in_v_proj_weight, transpose_b=True)

            if self.use_bias:
                k += self.in_k_proj_bias
                v += self.in_v_proj_bias

            # Reshape into (batch_size * num_heads, seq_len, head_dim)
            k = tf.reshape(k, [-1, bsz * self.num_heads, self.head_dim])
            k = tf.transpose(k, [1, 0, 2])
            v = tf.reshape(v, [-1, bsz * self.num_heads, self.head_dim])
            v = tf.transpose(v, [1, 0, 2])

        # Save key and value tensor to memory
        saved_kv_shape = [bsz, self.num_heads, -1, self.head_dim]
        self._set_saved_state(incremental_state, {
            'prev_key': tf.reshape(k, saved_kv_shape),
            'prev_value': tf.reshape(v, saved_kv_shape)
        })

        src_len = tf.shape(k)[1]

        # Compute attention weights also mask attn if needed
        attn_output_weights = tf.linalg.matmul(q, tf.transpose(k, [0, 2, 1]))
        # attn_output_weights_shape = [bsz * num_heads, tgt_len, src_len]

        # Don't attend to padding symbols
        key_padding_mask = \
            tf.expand_dims(tf.expand_dims(key_padding_mask, 1), 2)
        attn_output_weights = tf.reshape(
            attn_output_weights, [bsz, self.num_heads, tgt_len, src_len])
        attn_output_weights = attn_output_weights - self.INF * \
            tf.cast(key_padding_mask, attn_output_weights.dtype)
        attn_output_weights = tf.reshape(
            attn_output_weights, [bsz * self.num_heads, tgt_len, src_len])

        # Apply softmax to get attention distribution
        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=2)
        if training:
            attn_output_weights = \
                tf.nn.dropout(attn_output_weights, rate=self.attn_dropout)

        # Apply attention weights to value to get outputs
        attn_output = tf.matmul(attn_output_weights, v)

        # Reshape back into (seq_len, batch_size, hidden_dim)
        attn_output = tf.transpose(attn_output, [1, 0, 2])
        attn_output = tf.reshape(attn_output, [tgt_len, bsz, self.hidden_dim])

        # Apply final linear layer
        attn_output = tf.matmul(
            attn_output, self.out_proj_weight,
            transpose_b=True
        )
        if self.use_bias:
            attn_output += self.out_proj_bias

        if need_weights:
            attn_output_weights = tf.reshape(
                attn_output_weights, [bsz, self.num_heads, tgt_len, src_len])
            attn_output_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, incremental_state, attn_output_weights
        else:
            return attn_output, incremental_state


class IncrementalMultiheadAttention(tf.keras.layers.Layer):
    """
    For incremental self-attention in decoder

    inputs
        - query_input (tgt_len, bsz, hidden_dim)
        - key_input (src_len, bsz, hidden_dim)
        - value_input (src_len, bsz, hidden_dim)
        - attn_mask (tgt_len, src_len) : 1 at positions to mask, 0 at others
        - key_padding_mask (bsz, src_len) : 1 at positions to mask, 0 at others
        - incremental_state (dict)
        - new_state_order (bsz)
    outputs
        - attn_output (tgt_len, bsz, hidden_dim)
        - attn_output_weights (bsz, tgt_len, src_len) : requires need_weights
    """

    _key_postfix = 'attn_state'

    def __init__(
        self,
        hidden_dim: int, num_heads: int,
        use_bias: bool = True, attn_dropout: float = 0.0,
        **kwargs
    ):
        super(IncrementalMultiheadAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.attn_dropout = attn_dropout

        self.head_dim = self.hidden_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        assert self.head_dim * self.num_heads == self.hidden_dim, \
            'hidden_dim must be divisible by num_heads'

    def build(self, input_shapes):
        query_input_shape = input_shapes[0]
        input_dim = int(query_input_shape[2])

        # Input and output linear layer weights
        self.in_q_proj_weight = self.add_weight(
            name='in_q_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.in_k_proj_weight = self.add_weight(
            name='in_k_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.in_v_proj_weight = self.add_weight(
            name='in_v_proj_weight',
            shape=[input_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )

        self.out_proj_weight = self.add_weight(
            name='out_proj_weight',
            shape=[self.hidden_dim, self.hidden_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )

        # Input and output linear layer bias
        if self.use_bias:
            self.in_q_proj_bias = self.add_weight(
                name='in_q_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )
            self.in_k_proj_bias = self.add_weight(
                name='in_k_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )
            self.in_v_proj_bias = self.add_weight(
                name='in_v_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )

            self.out_proj_bias = self.add_weight(
                name='out_proj_bias',
                shape=[self.hidden_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )

        # A negative infinity will be multiplied to masks and added to
        # tensors so that after softmax, the areas masked will have value of 0
        # As inf is unstable in tf. ie. 0 * inf = nan
        # The max value of the model dtype will be used
        if isinstance(self.dtype, tf.DType):
            self.INF = self.dtype.max
        else:
            # isinstance(self.dtype, str):
            dtype = getattr(tf, self.dtype)
            self.INF = dtype.max

        super(IncrementalMultiheadAttention, self).build(input_shapes)

    def _get_saved_state(self, incremental_state):
        return incr_decoding_utils.get_state(
            module_instance=self,
            incremental_state=incremental_state,
            key_postfix=self._key_postfix
        ) or {}

    def _set_saved_state(self, incremental_state, value):
        incr_decoding_utils.set_state(
            module_instance=self,
            incremental_state=incremental_state,
            key_postfix=self._key_postfix,
            value=value
        )

    def call(self, inputs, training=None, need_weights=None):
        (
            query_input, key_input, value_input, attn_mask,
            incremental_state, new_state_order
        ) = inputs

        query_input_shape = tf.shape(query_input)
        tgt_len = query_input_shape[0]
        bsz = query_input_shape[1]

        # Get previous state
        saved_state = self._get_saved_state(incremental_state)

        # Compute query, key and value tensor
        q = tf.matmul(query_input, self.in_q_proj_weight, transpose_b=True)
        k = tf.matmul(key_input, self.in_k_proj_weight, transpose_b=True)
        v = tf.matmul(value_input, self.in_v_proj_weight, transpose_b=True)
        if self.use_bias:
            q += self.in_q_proj_bias
            k += self.in_k_proj_bias
            v += self.in_v_proj_bias
        q *= self.scaling

        # Reshape into (batch_size * num_heads, seq_len, head_dim)
        q = tf.reshape(q, [-1, bsz * self.num_heads, self.head_dim])
        q = tf.transpose(q, [1, 0, 2])
        k = tf.reshape(k, [-1, bsz * self.num_heads, self.head_dim])
        k = tf.transpose(k, [1, 0, 2])
        v = tf.reshape(v, [-1, bsz * self.num_heads, self.head_dim])
        v = tf.transpose(v, [1, 0, 2])

        # Make use of saved state
        if 'prev_key' in saved_state:
            prev_key = saved_state['prev_key']
            prev_value = saved_state['prev_value']

            prev_key = tf.gather(prev_key, new_state_order)
            prev_value = tf.gather(prev_value, new_state_order)

            required_kv_shape = [bsz * self.num_heads, -1, self.head_dim]
            prev_key = tf.reshape(prev_key, required_kv_shape)
            prev_value = tf.reshape(prev_value, required_kv_shape)

            k = tf.concat([prev_key, k], axis=1)
            v = tf.concat([prev_value, v], axis=1)

        saved_kv_shape = [bsz, self.num_heads, -1, self.head_dim]
        self._set_saved_state(incremental_state, {
            'prev_key': tf.reshape(k, saved_kv_shape),
            'prev_value': tf.reshape(v, saved_kv_shape)
        })

        src_len = tf.shape(k)[1]

        # Compute attention weights also mask attn if needed
        attn_output_weights = tf.linalg.matmul(q, tf.transpose(k, [0, 2, 1]))
        # attn_output_weights_shape = [bsz * num_heads, tgt_len, src_len]

        # Don't attend to masked positions
        attn_output_weights = attn_output_weights - self.INF * \
            tf.cast(attn_mask, attn_output_weights.dtype)

        # Apply softmax to get attention distribution
        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=2)
        if training:
            attn_output_weights = \
                tf.nn.dropout(attn_output_weights, rate=self.attn_dropout)

        # Apply attention weights to value to get outputs
        attn_output = tf.matmul(attn_output_weights, v)

        # Reshape back into (seq_len, batch_size, hidden_dim)
        attn_output = tf.transpose(attn_output, [1, 0, 2])
        attn_output = tf.reshape(attn_output, [tgt_len, bsz, self.hidden_dim])

        # Apply final linear layer
        attn_output = tf.matmul(
            attn_output, self.out_proj_weight,
            transpose_b=True
        )
        if self.use_bias:
            attn_output += self.out_proj_bias

        if need_weights:
            attn_output_weights = tf.reshape(
                attn_output_weights, [bsz, self.num_heads, tgt_len, src_len])
            attn_output_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, incremental_state, attn_output_weights
        else:
            return attn_output, incremental_state
