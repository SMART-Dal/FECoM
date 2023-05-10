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
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
env = gym.make('CartPole-v1')
seed = 42
custom_method(
tf.random.set_seed(seed), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.random.set_seed(*args)', method_object=None, object_signature=None, function_args=[eval('seed')], function_kwargs={})
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()
        self.common = custom_method(
        layers.Dense(num_hidden_units, activation='relu'), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='layers.Dense(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('num_hidden_units')], function_kwargs={'activation': eval('"relu"')})
        self.actor = custom_method(
        layers.Dense(num_actions), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='layers.Dense(*args)', method_object=None, object_signature=None, function_args=[eval('num_actions')], function_kwargs={})
        self.critic = custom_method(
        layers.Dense(1), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='layers.Dense(*args)', method_object=None, object_signature=None, function_args=[eval('1')], function_kwargs={})

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return (self.actor(x), self.critic(x))
num_actions = env.action_space.n
num_hidden_units = 128
model = ActorCritic(num_actions, num_hidden_units)

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""
    (state, reward, done, truncated, info) = env.step(action)
    return (custom_method(
    state.astype(np.float32), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.astype(*args)', method_object=eval('state'), object_signature=None, function_args=[eval('np.float32')], function_kwargs={}, custom_class=None), np.array(reward, np.int32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return custom_method(
    tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32]), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.numpy_function(*args)', method_object=None, object_signature=None, function_args=[eval('env_step'), eval('[action]'), eval('[tf.float32, tf.int32, tf.int32]')], function_kwargs={})

def run_episode(initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""
    action_probs = custom_method(
    tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.TensorArray(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'dtype': eval('tf.float32'), 'size': eval('0'), 'dynamic_size': eval('True')})
    values = custom_method(
    tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.TensorArray(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'dtype': eval('tf.float32'), 'size': eval('0'), 'dynamic_size': eval('True')})
    rewards = custom_method(
    tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.TensorArray(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'dtype': eval('tf.int32'), 'size': eval('0'), 'dynamic_size': eval('True')})
    initial_state_shape = initial_state.shape
    state = initial_state
    for t in custom_method(
    tf.range(max_steps), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.range(*args)', method_object=None, object_signature=None, function_args=[eval('max_steps')], function_kwargs={}):
        state = custom_method(
        tf.expand_dims(state, 0), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('state'), eval('0')], function_kwargs={})
        (action_logits_t, value) = custom_method(
        model(state), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('state')], function_kwargs={}, custom_class='class ActorCritic(tf.keras.Model):\n  """Combined actor-critic network."""\n\n  def __init__(\n      self, \n      num_actions: int, \n      num_hidden_units: int):\n    """Initialize."""\n    super().__init__()\n\n    self.common = layers.Dense(num_hidden_units, activation="relu")\n    self.actor = layers.Dense(num_actions)\n    self.critic = layers.Dense(1)\n\n  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:\n    x = self.common(inputs)\n    return self.actor(x), self.critic(x)')
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = custom_method(
        tf.nn.softmax(action_logits_t), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[eval('action_logits_t')], function_kwargs={})
        values = custom_method(
        values.write(t, tf.squeeze(value)), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.write(*args)', method_object=eval('values'), object_signature=None, function_args=[eval('t'), eval('tf.squeeze(value)')], function_kwargs={}, custom_class=None)
        action_probs = custom_method(
        action_probs.write(t, action_probs_t[0, action]), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.write(*args)', method_object=eval('action_probs'), object_signature=None, function_args=[eval('t'), eval('action_probs_t[0, action]')], function_kwargs={}, custom_class=None)
        (state, reward, done) = tf_env_step(action)
        custom_method(
        state.set_shape(initial_state_shape), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.set_shape(*args)', method_object=eval('state'), object_signature=None, function_args=[eval('initial_state_shape')], function_kwargs={}, custom_class=None)
        rewards = custom_method(
        rewards.write(t, reward), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.write(*args)', method_object=eval('rewards'), object_signature=None, function_args=[eval('t'), eval('reward')], function_kwargs={}, custom_class=None)
        if custom_method(
        tf.cast(done, tf.bool), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('done'), eval('tf.bool')], function_kwargs={}):
            break
    action_probs = custom_method(
    action_probs.stack(), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.stack()', method_object=eval('action_probs'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    values = custom_method(
    values.stack(), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.stack()', method_object=eval('values'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    rewards = custom_method(
    rewards.stack(), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.stack()', method_object=eval('rewards'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    return (action_probs, values, rewards)

def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool=True) -> tf.Tensor:
    """Compute expected returns per timestep."""
    n = tf.shape(rewards)[0]
    returns = custom_method(
    tf.TensorArray(dtype=tf.float32, size=n), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.TensorArray(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'dtype': eval('tf.float32'), 'size': eval('n')})
    rewards = custom_method(
    tf.cast(rewards[::-1], dtype=tf.float32), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.cast(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('rewards[::-1]')], function_kwargs={'dtype': eval('tf.float32')})
    discounted_sum = custom_method(
    tf.constant(0.0), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('0.0')], function_kwargs={})
    discounted_sum_shape = discounted_sum.shape
    for i in custom_method(
    tf.range(n), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.range(*args)', method_object=None, object_signature=None, function_args=[eval('n')], function_kwargs={}):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        custom_method(
        discounted_sum.set_shape(discounted_sum_shape), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.set_shape(*args)', method_object=eval('discounted_sum'), object_signature=None, function_args=[eval('discounted_sum_shape')], function_kwargs={}, custom_class=None)
        returns = custom_method(
        returns.write(i, discounted_sum), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.write(*args)', method_object=eval('returns'), object_signature=None, function_args=[eval('i'), eval('discounted_sum')], function_kwargs={}, custom_class=None)
    returns = returns.stack()[::-1]
    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)
    return returns
huber_loss = custom_method(
tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.keras.losses.Huber(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'reduction': eval('tf.keras.losses.Reduction.SUM')})

def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""
    advantage = returns - values
    action_log_probs = custom_method(
    tf.math.log(action_probs), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.math.log(*args)', method_object=None, object_signature=None, function_args=[eval('action_probs')], function_kwargs={})
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    critic_loss = custom_method(
    huber_loss(values, returns), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj(*args)', method_object=eval('huber_loss'), object_signature=None, function_args=[eval('values'), eval('returns')], function_kwargs={}, custom_class=None)
    return actor_loss + critic_loss
optimizer = custom_method(
tf.keras.optimizers.Adam(learning_rate=0.01), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('0.01')})

@tf.function
def train_step(initial_state: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, gamma: float, max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""
    with custom_method(
    tf.GradientTape(), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}) as tape:
        (action_probs, values, rewards) = run_episode(initial_state, model, max_steps_per_episode)
        returns = get_expected_return(rewards, gamma)
        (action_probs, values, returns) = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
        loss = compute_loss(action_probs, values, returns)
    grads = tape.gradient(loss, model.trainable_variables)
    custom_method(
    optimizer.apply_gradients(zip(grads, model.trainable_variables)), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), object_signature=None, function_args=[eval('zip(grads, model.trainable_variables)')], function_kwargs={}, custom_class=None)
    episode_reward = custom_method(
    tf.math.reduce_sum(rewards), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.math.reduce_sum(*args)', method_object=None, object_signature=None, function_args=[eval('rewards')], function_kwargs={})
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
    initial_state = custom_method(
    tf.constant(initial_state, dtype=tf.float32), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.constant(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('initial_state')], function_kwargs={'dtype': eval('tf.float32')})
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
    state = custom_method(
    tf.constant(state, dtype=tf.float32), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.constant(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('state')], function_kwargs={'dtype': eval('tf.float32')})
    screen = env.render()
    images = [Image.fromarray(screen)]
    for i in range(1, max_steps + 1):
        state = custom_method(
        tf.expand_dims(state, 0), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('state'), eval('0')], function_kwargs={})
        (action_probs, _) = custom_method(
        model(state), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('state')], function_kwargs={}, custom_class='class ActorCritic(tf.keras.Model):\n  """Combined actor-critic network."""\n\n  def __init__(\n      self, \n      num_actions: int, \n      num_hidden_units: int):\n    """Initialize."""\n    super().__init__()\n\n    self.common = layers.Dense(num_hidden_units, activation="relu")\n    self.actor = layers.Dense(num_actions)\n    self.critic = layers.Dense(1)\n\n  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:\n    x = self.common(inputs)\n    return self.actor(x), self.critic(x)')
        action = np.argmax(np.squeeze(action_probs))
        (state, reward, done, truncated, info) = env.step(action)
        state = custom_method(
        tf.constant(state, dtype=tf.float32), imports='from PIL import Image;import collections;from tensorflow.keras import layers;from IPython import display as ipythondisplay;import gym;import tqdm;from typing import Any, List, Sequence, Tuple;from matplotlib import pyplot as plt;import statistics;import tensorflow as tf;import tensorflow_docs.vis.embed as embed;import numpy as np', function_to_run='tf.constant(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('state')], function_kwargs={'dtype': eval('tf.float32')})
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
