import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import os
from pathlib import Path
import dill as pickle
import sys
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
dftrain.head()
dftrain.describe()
(dftrain.shape[0], dfeval.shape[0])
dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):

    def input_function():
        custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(dict(data_df), label_df)')], function_kwargs={})
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='obj.shuffle(*args)', method_object=eval('ds'), object_signature=None, function_args=[eval('1000')], function_kwargs={}, custom_class=None)
            ds = ds.shuffle(1000)
        custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='obj.batch(batch_size).repeat(*args)', method_object=eval('ds'), object_signature=None, function_args=[eval('num_epochs')], function_kwargs={}, custom_class=None)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
ds = make_input_fn(dftrain, y_train, batch_size=10)()
for (feature_batch, label_batch) in ds.take(1):
    print('Some feature keys:', list(feature_batch.keys()))
    print()
    print('A batch of class:', feature_batch['class'].numpy())
    print()
    print('A batch of Labels:', label_batch.numpy())
age_column = feature_columns[7]
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()
gender_column = feature_columns[0]
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='tf.estimator.LinearClassifier(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'feature_columns': eval('feature_columns')})
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='obj.train(*args)', method_object=eval('linear_est'), object_signature=None, function_args=[eval('train_input_fn')], function_kwargs={}, custom_class=None)
linear_est.train(train_input_fn)
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='obj.evaluate(*args)', method_object=eval('linear_est'), object_signature=None, function_args=[eval('eval_input_fn')], function_kwargs={}, custom_class=None)
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(result)
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='tf.feature_column.crossed_column(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("['age', 'sex']")], function_kwargs={'hash_bucket_size': eval('100')})
age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
derived_feature_columns = [age_x_gender]
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='tf.estimator.LinearClassifier(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'feature_columns': eval('feature_columns+derived_feature_columns')})
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns + derived_feature_columns)
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='obj.train(*args)', method_object=eval('linear_est'), object_signature=None, function_args=[eval('train_input_fn')], function_kwargs={}, custom_class=None)
linear_est.train(train_input_fn)
custom_method(imports='import pandas as pd;import sys;import os;import tensorflow.compat.v2.feature_column as fc;from IPython.display import clear_output;from six.moves import urllib;import tensorflow as tf;from matplotlib import pyplot as plt;from sklearn.metrics import roc_curve;import matplotlib.pyplot as plt;import numpy as np', function_to_run='obj.evaluate(*args)', method_object=eval('linear_est'), object_signature=None, function_args=[eval('eval_input_fn')], function_kwargs={}, custom_class=None)
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(result)
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
probs.plot(kind='hist', bins=20, title='predicted probabilities')
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
(fpr, tpr, _) = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0)
plt.ylim(0)
