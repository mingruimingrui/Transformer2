This readme will currently be used to collate some anti-patterns used in this
codebase. Usually anti-patterns aren't good to have but the anti-patterns
listed here are used to make the code work. So a README is needed to explain
how they function.

## Variable shape control flow

In `tf.range` loops, tensors are needed to be constant shaped, ie.
the shape of a named tensor needs to have the same number of dimensions and
dimension sizes throughout the entirety of the loop.

An anti-pattern is used to overcome this limitation of `tf2.0`. The way this
package does it is with an object state as shown in
`varied_shape_control_flow.py`.

```
import tensorflow as tf


class MyModule(object):
    def __init__(self):
        self.x = None

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
```

The variable shaped tensor is `module.x` on each step of `tf.range`, the shape
of `module.x` changes.

This anti-pattern can be observed in `CachedMultiheadAttention` and
`IncrementalMultiheadAttention` to store the previous key and value tensor
states.
