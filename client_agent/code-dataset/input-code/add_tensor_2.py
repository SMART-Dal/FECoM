import tensorflow as tf

# Enable eager execution
tf.executing_eagerly()

# Define two tensors
a = tf.constant([1, 2, 3, 4])
c = tf.constant([20, 20, 30, 40])

# Add the two tensors
c = a + b

# Print the result
print(c)
