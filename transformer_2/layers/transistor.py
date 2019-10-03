import tensorflow as tf


class Transistor(tf.keras.layers.Layer):
    """
    Transistor layer that keeps embedding size constant
    """

    def __init__(
        self,
        inner_dim: int,
        activation='relu',
        use_bias: bool = True,
        activation_dropout: float = 0.0,
        **kwargs
    ):
        super(Transistor, self).__init__(**kwargs)
        self.inner_dim = inner_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.activation_dropout = activation_dropout

    def build(self, input_shape):
        last_dim = int(input_shape[-1])

        self.kernel_0 = self.add_weight(
            name='kernel_0',
            # shape=[last_dim, self.inner_dim],
            shape=[self.inner_dim, last_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.kernel_1 = self.add_weight(
            name='kernel_1',
            # shape=[self.inner_dim, last_dim],
            shape=[last_dim, self.inner_dim],
            dtype=self.dtype,
            initializer=tf.keras.initializers.GlorotUniform()
        )

        if self.use_bias:
            self.bias_0 = self.add_weight(
                name='bias_0',
                shape=[self.inner_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )
            self.bias_1 = self.add_weight(
                name='bias_1',
                shape=[last_dim],
                dtype=self.dtype,
                initializer=tf.keras.initializers.Constant(0)
            )

        super(Transistor, self).build(input_shape)

    def call(self, x, training=None):
        x = tf.matmul(x, self.kernel_0, transpose_b=True)
        if self.use_bias:
            x += self.bias_0
        if self.activation is not None:
            x = self.activation(x)
        x = tf.nn.dropout(x, rate=self.activation_dropout) if training else x
        x = tf.matmul(x, self.kernel_1, transpose_b=True)
        if self.use_bias:
            x += self.bias_1
        return x
