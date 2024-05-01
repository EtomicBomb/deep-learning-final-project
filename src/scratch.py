import tensorflow as tf

""" Baby file used to unit-test development on mini-matrices to verify the
    tensorflow functions do what I expect
"""


# x = tf.random.normal((2,2))
x = tf.random.uniform(shape=(2, 1), minval=0, maxval=10, dtype=tf.int32)
print(f"x: {x}")

x = tf.tile(x, [1, 3])
print(f"x after tiling: {x}")