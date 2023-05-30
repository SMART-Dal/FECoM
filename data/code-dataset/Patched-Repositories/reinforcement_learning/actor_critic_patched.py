import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import os
from pathlib import Path
import dill as pickle
import sys
import numpy as np
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
import json
current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
skip_calls_file_path = EXPERIMENT_FILE_PATH.parent / 'skip_calls.json'
if skip_calls_file_path.exists():
    with open(skip_calls_file_path, 'r') as f:
        skip_calls = json.load(f)
else:
    skip_calls = []
    with open(skip_calls_file_path, 'w') as f:
        json.dump(skip_calls, f)

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    if skip_calls is not None and any((call['function_to_run'] == function_to_run and np.array_equal(call['function_args'], function_args) and (call['function_kwargs'] == function_kwargs) for call in skip_calls)):
        print('skipping call: ', function_to_run)
        return
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    if result is not None and isinstance(result, dict) and (len(result) == 1):
        energy_data = next(iter(result.values()))
        if skip_calls is not None and 'start_time_perf' in energy_data['times'] and ('end_time_perf' in energy_data['times']) and ('start_time_nvidia' in energy_data['times']) and ('end_time_nvidia' in energy_data['times']) and (energy_data['times']['start_time_perf'] == energy_data['times']['end_time_perf']) and (energy_data['times']['start_time_nvidia'] == energy_data['times']['end_time_nvidia']):
            call_to_skip = {'function_to_run': function_to_run, 'function_args': function_args, 'function_kwargs': function_kwargs}
            try:
                json.dumps(call_to_skip)
                if call_to_skip not in skip_calls:
                    skip_calls.append(call_to_skip)
                    with open(skip_calls_file_path, 'w') as f:
                        json.dump(skip_calls, f)
                    print('skipping call added, current list is: ', skip_calls)
                else:
                    print('Skipping call already exists.')
            except TypeError:
                print('Ignore: Skipping call is not JSON serializable, skipping append and dump.')
    else:
        print('Invalid dictionary object or does not have one key-value pair.')
env = gym.make('CartPole-v1')
seed = 42
custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.random.set_seed(*args)', method_object=None, object_signature=None, function_args=[eval('seed')], function_kwargs={})
tf.random.set_seed(seed)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='layers.Dense(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('num_hidden_units')], function_kwargs={'activation': eval('"relu"')})
        self.common = layers.Dense(num_hidden_units, activation='relu')
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='layers.Dense(*args)', method_object=None, object_signature=None, function_args=[eval('num_actions')], function_kwargs={})
        self.actor = layers.Dense(num_actions)
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='layers.Dense(*args)', method_object=None, object_signature=None, function_args=[eval('1')], function_kwargs={})
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return (self.actor(x), self.critic(x))
num_actions = env.action_space.n
num_hidden_units = 128
model = ActorCritic(num_actions, num_hidden_units)

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""
    (state, reward, done, truncated, info) = env.step(action)
    return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])

def run_episode(initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.TensorArray(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'dtype': eval('tf.float32'), 'size': eval('0'), 'dynamic_size': eval('True')})
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.TensorArray(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'dtype': eval('tf.float32'), 'size': eval('0'), 'dynamic_size': eval('True')})
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.TensorArray(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'dtype': eval('tf.int32'), 'size': eval('0'), 'dynamic_size': eval('True')})
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    initial_state_shape = initial_state.shape
    state = initial_state
    for t in tf.range(max_steps):
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('state'), eval('0')], function_kwargs={})
        state = tf.expand_dims(state, 0)
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('state')], function_kwargs={}, custom_class='class ActorCritic(tf.keras.Model):\n  """Combined actor-critic network."""\n\n  def __init__(\n      self, \n      num_actions: int, \n      num_hidden_units: int):\n    """Initialize."""\n    super().__init__()\n\n    self.common = layers.Dense(num_hidden_units, activation="relu")\n    self.actor = layers.Dense(num_actions)\n    self.critic = layers.Dense(1)\n\n  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:\n    x = self.common(inputs)\n    return self.actor(x), self.critic(x)')
        (action_logits_t, value) = model(state)
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[eval('action_logits_t')], function_kwargs={})
        action_probs_t = tf.nn.softmax(action_logits_t)
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.write(*args)', method_object=eval('values'), object_signature=None, function_args=[eval('t'), eval('tf.squeeze(value)')], function_kwargs={}, custom_class=None)
        values = values.write(t, tf.squeeze(value))
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.write(*args)', method_object=eval('action_probs'), object_signature=None, function_args=[eval('t'), eval('action_probs_t[0, action]')], function_kwargs={}, custom_class=None)
        action_probs = action_probs.write(t, action_probs_t[0, action])
        (state, reward, done) = tf_env_step(action)
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.set_shape(*args)', method_object=eval('state'), object_signature=None, function_args=[eval('initial_state_shape')], function_kwargs={}, custom_class=None)
        state.set_shape(initial_state_shape)
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.write(*args)', method_object=eval('rewards'), object_signature=None, function_args=[eval('t'), eval('reward')], function_kwargs={}, custom_class=None)
        rewards = rewards.write(t, reward)
        if tf.cast(done, tf.bool):
            break
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.stack()', method_object=eval('action_probs'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    action_probs = action_probs.stack()
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.stack()', method_object=eval('values'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    values = values.stack()
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.stack()', method_object=eval('rewards'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    rewards = rewards.stack()
    return (action_probs, values, rewards)

def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool=True) -> tf.Tensor:
    """Compute expected returns per timestep."""
    n = tf.shape(rewards)[0]
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.TensorArray(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'dtype': eval('tf.float32'), 'size': eval('n')})
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.cast(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('rewards[::-1]')], function_kwargs={'dtype': eval('tf.float32')})
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('0.0')], function_kwargs={})
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.set_shape(*args)', method_object=eval('discounted_sum'), object_signature=None, function_args=[eval('discounted_sum_shape')], function_kwargs={}, custom_class=None)
        discounted_sum.set_shape(discounted_sum_shape)
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.write(*args)', method_object=eval('returns'), object_signature=None, function_args=[eval('i'), eval('discounted_sum')], function_kwargs={}, custom_class=None)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]
    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)
    return returns
custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.keras.losses.Huber(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'reduction': eval('tf.keras.losses.Reduction.SUM')})
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""
    advantage = returns - values
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.math.log(*args)', method_object=None, object_signature=None, function_args=[eval('action_probs')], function_kwargs={})
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj(*args)', method_object=eval('huber_loss'), object_signature=None, function_args=[eval('values'), eval('returns')], function_kwargs={}, custom_class=None)
    critic_loss = huber_loss(values, returns)
    return actor_loss + critic_loss
custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('0.01')})
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function
def train_step(initial_state: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, gamma: float, max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""
    with tf.GradientTape() as tape:
        (action_probs, values, rewards) = run_episode(initial_state, model, max_steps_per_episode)
        returns = get_expected_return(rewards, gamma)
        (action_probs, values, returns) = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
        loss = compute_loss(action_probs, values, returns)
    grads = tape.gradient(loss, model.trainable_variables)
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), object_signature=None, function_args=[eval('zip(grads, model.trainable_variables)')], function_kwargs={}, custom_class=None)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.math.reduce_sum(*args)', method_object=None, object_signature=None, function_args=[eval('rewards')], function_kwargs={})
    episode_reward = tf.math.reduce_sum(rewards)
    return episode_reward
min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 500
reward_threshold = 475
running_reward = 0
gamma = 0.99
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
t = tqdm.trange(max_episodes)
for i in t:
    (initial_state, info) = env.reset()
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.constant(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('initial_state')], function_kwargs={'dtype': eval('tf.float32')})
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))
    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)
    t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
    if i % 10 == 0:
        pass
    if running_reward > reward_threshold and i >= min_episodes_criterion:
        break
print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
from IPython import display as ipythondisplay
from PIL import Image
render_env = gym.make('CartPole-v1', render_mode='rgb_array')

def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int):
    (state, info) = env.reset()
    custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.constant(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('state')], function_kwargs={'dtype': eval('tf.float32')})
    state = tf.constant(state, dtype=tf.float32)
    screen = env.render()
    images = [Image.fromarray(screen)]
    for i in range(1, max_steps + 1):
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('state'), eval('0')], function_kwargs={})
        state = tf.expand_dims(state, 0)
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('state')], function_kwargs={}, custom_class='class ActorCritic(tf.keras.Model):\n  """Combined actor-critic network."""\n\n  def __init__(\n      self, \n      num_actions: int, \n      num_hidden_units: int):\n    """Initialize."""\n    super().__init__()\n\n    self.common = layers.Dense(num_hidden_units, activation="relu")\n    self.actor = layers.Dense(num_actions)\n    self.critic = layers.Dense(1)\n\n  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:\n    x = self.common(inputs)\n    return self.actor(x), self.critic(x)')
        (action_probs, _) = model(state)
        action = np.argmax(np.squeeze(action_probs))
        (state, reward, done, truncated, info) = env.step(action)
        custom_method(imports='from matplotlib import pyplot as plt;import collections;from tensorflow.keras import layers;import tensorflow_docs.vis.embed as embed;from typing import Any, List, Sequence, Tuple;import tensorflow as tf;from PIL import Image;import tqdm;import numpy as np;from IPython import display as ipythondisplay;import statistics;import gym', function_to_run='tf.constant(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('state')], function_kwargs={'dtype': eval('tf.float32')})
        state = tf.constant(state, dtype=tf.float32)
        if i % 10 == 0:
            screen = env.render()
            images.append(Image.fromarray(screen))
        if done:
            break
    return images
images = render_episode(render_env, model, max_steps_per_episode)
image_file = 'cartpole-v1.gif'
images[0].save(image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
import tensorflow_docs.vis.embed as embed
embed.embed_file(image_file)
