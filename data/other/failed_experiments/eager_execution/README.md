Data in this directory was run obtained with the [run_eagerly option](https://discuss.tensorflow.org/t/what-does-the-run-eagerly-parameter-in-model-compile-do/1924) set to `True` in `model.compile`:  
```
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                run_eagerly=True)
```
This was done in an attempt to debug a server error, which was later found to be unrelated to this setting. It resulted in significantly higher energy consumption than the default, potentially worth some investigation!