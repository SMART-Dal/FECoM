import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiments import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
env = gym.make('CartPole-v1')
seed = 42
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.set_seed()')
tf.random.set_seed(seed)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.set_seed()', method_object=None, function_args=[seed], function_kwargs=None)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense()')
        self.common = layers.Dense(num_hidden_units, activation='relu')
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense()', method_object=None, function_args=[num_hidden_units], function_kwargs={'activation': 'relu'})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense()')
        self.actor = layers.Dense(num_actions)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense()', method_object=None, function_args=[num_actions], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense()')
        self.critic = layers.Dense(1)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense()', method_object=None, function_args=[1], function_kwargs=None)

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
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray()')
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray()', method_object=None, function_args=None, function_kwargs={'dtype': tf.float32, 'size': 0, 'dynamic_size': True})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray()')
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray()', method_object=None, function_args=None, function_kwargs={'dtype': tf.float32, 'size': 0, 'dynamic_size': True})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray()')
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray()', method_object=None, function_args=None, function_kwargs={'dtype': tf.int32, 'size': 0, 'dynamic_size': True})
    initial_state_shape = initial_state.shape
    state = initial_state
    for t in tf.range(max_steps):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.expand_dims()')
        state = tf.expand_dims(state, 0)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.expand_dims()', method_object=None, function_args=[state, 0], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model()')
        (action_logits_t, value) = model(state)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model()', method_object=model, function_args=[state], function_kwargs=None)
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nn.softmax()')
        action_probs_t = tf.nn.softmax(action_logits_t)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nn.softmax()', method_object=None, function_args=[action_logits_t], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write()')
        values = values.write(t, tf.squeeze(value))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write()', method_object=values, function_args=[t, tf.squeeze(value)], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write()')
        action_probs = action_probs.write(t, action_probs_t[0, action])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write()', method_object=action_probs, function_args=[t, action_probs_t[0, action]], function_kwargs=None)
        (state, reward, done) = tf_env_step(action)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.expand_dims.set_shape()')
        state.set_shape(initial_state_shape)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.expand_dims.set_shape()', method_object=state, function_args=[initial_state_shape], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write()')
        rewards = rewards.write(t, reward)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write()', method_object=rewards, function_args=[t, reward], function_kwargs=None)
        if tf.cast(done, tf.bool):
            break
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write.stack()')
    action_probs = action_probs.stack()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write.stack()', method_object=action_probs, function_args=None, function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write.stack()')
    values = values.stack()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write.stack()', method_object=values, function_args=None, function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write.stack()')
    rewards = rewards.stack()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write.stack()', method_object=rewards, function_args=None, function_kwargs=None)
    return (action_probs, values, rewards)

def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool=True) -> tf.Tensor:
    """Compute expected returns per timestep."""
    n = tf.shape(rewards)[0]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray()')
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray()', method_object=None, function_args=None, function_kwargs={'dtype': tf.float32, 'size': n})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.cast()')
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.cast()', method_object=None, function_args=[rewards[::-1]], function_kwargs={'dtype': tf.float32})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant()')
    discounted_sum = tf.constant(0.0)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant()', method_object=None, function_args=[0.0], function_kwargs=None)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant.set_shape()')
        discounted_sum.set_shape(discounted_sum_shape)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant.set_shape()', method_object=discounted_sum, function_args=[discounted_sum_shape], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write()')
        returns = returns.write(i, discounted_sum)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.TensorArray.write()', method_object=returns, function_args=[i, discounted_sum], function_kwargs=None)
    returns = returns.stack()[::-1]
    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)
    return returns
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.Huber()')
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.Huber()', method_object=None, function_args=None, function_kwargs={'reduction': tf.keras.losses.Reduction.SUM})

def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""
    advantage = returns - values
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.math.log()')
    action_log_probs = tf.math.log(action_probs)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.math.log()', method_object=None, function_args=[action_probs], function_kwargs=None)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.Huber()')
    critic_loss = huber_loss(values, returns)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.Huber()', method_object=huber_loss, function_args=[values, returns], function_kwargs=None)
    return actor_loss + critic_loss
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.Adam()')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.Adam()', method_object=None, function_args=None, function_kwargs={'learning_rate': 0.01})

@tf.function
def train_step(initial_state: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, gamma: float, max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""
    with tf.GradientTape() as tape:
        (action_probs, values, rewards) = run_episode(initial_state, model, max_steps_per_episode)
        returns = get_expected_return(rewards, gamma)
        (action_probs, values, returns) = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
        loss = compute_loss(action_probs, values, returns)
    grads = tape.gradient(loss, model.trainable_variables)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.Adam.apply_gradients()')
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.Adam.apply_gradients()', method_object=optimizer, function_args=[zip(grads, model.trainable_variables)], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.math.reduce_sum()')
    episode_reward = tf.math.reduce_sum(rewards)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.math.reduce_sum()', method_object=None, function_args=[rewards], function_kwargs=None)
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
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant()')
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant()', method_object=None, function_args=[initial_state], function_kwargs={'dtype': tf.float32})
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
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant()')
    state = tf.constant(state, dtype=tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant()', method_object=None, function_args=[state], function_kwargs={'dtype': tf.float32})
    screen = env.render()
    images = [Image.fromarray(screen)]
    for i in range(1, max_steps + 1):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.expand_dims()')
        state = tf.expand_dims(state, 0)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.expand_dims()', method_object=None, function_args=[state, 0], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model()')
        (action_probs, _) = model(state)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model()', method_object=model, function_args=[state], function_kwargs=None)
        action = np.argmax(np.squeeze(action_probs))
        (state, reward, done, truncated, info) = env.step(action)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant()')
        state = tf.constant(state, dtype=tf.float32)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.constant()', method_object=None, function_args=[state], function_kwargs={'dtype': tf.float32})
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
