import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.PROJECT_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
env = gym.make('CartPole-v1')
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()
        self.common = layers.Dense(num_hidden_units, activation='relu')
        self.actor = layers.Dense(num_actions)
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
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    initial_state_shape = initial_state.shape
    state = initial_state
    for t in tf.range(max_steps):
        state = tf.expand_dims(state, 0)
        (action_logits_t, value) = model(state)
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        values = values.write(t, tf.squeeze(value))
        action_probs = action_probs.write(t, action_probs_t[0, action])
        (state, reward, done) = tf_env_step(action)
        state.set_shape(initial_state_shape)
        rewards = rewards.write(t, reward)
        if tf.cast(done, tf.bool):
            break
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    return (action_probs, values, rewards)

def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool=True) -> tf.Tensor:
    """Compute expected returns per timestep."""
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]
    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)
    return returns
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""
    advantage = returns - values
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    critic_loss = huber_loss(values, returns)
    return actor_loss + critic_loss
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
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
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
    state = tf.constant(state, dtype=tf.float32)
    screen = env.render()
    images = [Image.fromarray(screen)]
    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        (action_probs, _) = model(state)
        action = np.argmax(np.squeeze(action_probs))
        (state, reward, done, truncated, info) = env.step(action)
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
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
