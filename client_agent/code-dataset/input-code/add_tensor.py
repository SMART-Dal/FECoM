import tensorflow as tf


tf.compat.v1.disable_eager_execution()

const1 = tf.constant([[1,2,3], [1,2,3]])
const2 = tf.constant([[3,4,5], [3,4,5]])

result = tf.add(const1, const2)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    output = sess.run(result)
    print(output)