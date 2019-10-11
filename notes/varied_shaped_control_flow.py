import tensorflow as tf


class MyModule(object):
    def __init__(self):
        self.x = None

    def set(self, x):
        self.x = x

    def get(self):
        return self.x


module = MyModule()


@tf.function
def foo():
    for i in tf.range(10):
        module.set(tf.random.normal([i]))
        module_x_shape = tf.shape(module.get())
        tf.print('Visualizing shape of module.x: {}'.format(module_x_shape))


print('Example of having a variable shape tensor in a tf.range loop.')
print('A variable module.x is created.')
print('Observe that at each run of the loop, it\'s shape changes.')
foo()
