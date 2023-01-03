import tensorflow as tf

# source: https://www.tensorflow.org/tutorials/quickstart/beginner

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# return vector of logits (scores), one for each class
predictions = model(x_train[:1]).numpy()

# converts logits to probabilities
tf.nn.softmax(predictions).numpy()

# loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# should give probabilities close to random
loss_fn(y_train[:1], predictions).numpy()

# configure & compile model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# training
model.fit(x_train, y_train, epochs=5)

# testing
model.evaluate(x_test,  y_test, verbose=2)

# wrap trained model and attach softmax to return probabilities
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])