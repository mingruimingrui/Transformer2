import tensorflow as tf


class MyModule(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyModule, self).__init__(**kwargs)
        self.x = None

    def build(self, input_shapes):
        super(MyModule, self).build(input_shapes)

    def call(self, inputs):
        return inputs

    def set(self, x):
        self.x = x

    def get(self):
        return self.x


module = MyModule()


@tf.function
def foo():
    for i in tf.range(1, 10):
        x = tf.range(i)
        module.set(x)
        module_x_shape = tf.shape(module.get())
        tf.print('Visualizing shape of module.x: ', module_x_shape)


print('Example of having a variable shape tensor in a tf.range loop.')
print('A variable module.x is created.')
print('Observe that at each run of the loop, it\'s shape changes.')
foo()

print('Observe that the variable shaped tensor exists only in tf.range')
print(module.x)
