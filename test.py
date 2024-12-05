import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

# Simple tensor operation to check GPU usage
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3 matrix
b = tf.constant([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])  # 3x2 matrix, compatible with 'a' for multiplication

c = tf.matmul(a, b)  # Should now work, resulting in a 2x2 matrix

print(c)
