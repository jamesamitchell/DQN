import haiku as hk
import jax.numpy as jnp
import jax
import numpy as np
import optax
from typing import NamedTuple, Tuple
import argparse
import gym
from typing import NamedTuple
import matplotlib.pyplot as plt
import pickle as pkl

def parse_args():
    """
    By defining all of the parameters using an argument parser
    we can easily change the parameters of our experiment in one place
    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.env_id = "CartPole-v1"
    args.seed = 1
    args.total_steps = 1000
    args.training_start = 100
    args.minibatch_size = 32
    args.replay_memory_size = 1000
    args.q_network_update_frequency = 10
    args.target_network_update_frequency = 100
    args.gamma = 0.99
    args.epsilon = 0.95
    args.learning_rate = 1e-3
    return args

def create_env(args: argparse.Namespace):
    """
    Simple function to create our environment
    """
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env

class TrainingState(NamedTuple):
    q_params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState

class Batch(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray

class ExperienceReplay:
    def __init__(self, size: int, env):
        """
        Initialises the arrays to store the observations.
        The shape of the arrays is dependant on the environment.
        This is enforced in the way the shape is defined relative to environment properties.
        """
        self.obs = np.zeros((size,) + env.observation_space.shape, dtype=np.float32)
        self.next_obs = np.zeros((size,) + env.observation_space.shape, dtype=np.float32)
        self.actions = np.zeros((size,) + env.action_space.shape, dtype=np.float32)
        self.rewards = np.zeros((size), dtype=np.float32)
        self.dones = np.zeros((size), dtype=np.float32)
        self.size = size
        self.pos = 0
        self.full = False
    
    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray):
        """
        Populates the empty arrays.
        """
        if len(obs) == 2:
            obs = obs[0]
        self.obs[self.pos] = obs.copy()
        self.next_obs[self.pos] = next_obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        # If the memory is full it starts refilling
        if self.pos == self.size:
            self.pos = 0
            self.full = True  
    
    def sample(self, size: int) -> Batch:
        """
        Randomly generates episodes to train on.
        jax.device_put puts the data onto the device, 
        making it available to use.
        """
        length = len(self.obs) if self.full else self.pos
        # Generates a random list of indicies to be used to extract the data
        indices = np.random.randint(length, size=size)
        # Collects the data and puts onto the device
        obs = jax.device_put(self.obs[indices])
        next_obs = jax.device_put(self.next_obs[indices])
        actions = jax.device_put(self.actions[indices])
        rewards = jax.device_put(self.rewards[indices])
        dones = jax.device_put(self.dones[indices])
        return Batch(obs, next_obs, actions, rewards, dones)

class Network(hk.Module):

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def __call__(self, x:jnp.ndarray) -> jnp.ndarray:

        net = hk.Sequential([
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(self.num_actions)
        ])
        return net(x)
   
@jax.jit
def policy(state:TrainingState, obs:jnp.ndarray) -> jnp.ndarray:
    q_values = q_network.apply(state.q_params, obs)
    print(q_values)
    action = jnp.argmax(q_values)
    print(action)
    return action

@jax.jit
def loss_fn(q_params:hk.Params, target_params:hk.Params, batch:Batch) -> jnp.ndarray:
    target_values = jax.lax.stop_gradient(target_network.apply(target_params, batch.next_obs))
    q_values = q_network.apply(q_params, batch.obs)
    rewards = batch.rewards + args.gamma * jnp.max(target_values) * (1 - batch.dones)
    loss = jnp.mean(jnp.square(rewards - jnp.max(q_values)))
    return loss

@jax.jit
def update(state:TrainingState, batch:Batch) -> Tuple[TrainingState, jnp.ndarray]:
    loss, gradients = jax.value_and_grad(loss_fn)(state.q_params, state.target_params, batch)
    updates, opt_state = optimizer.update(state.q_params, state.opt_state)
    q_params = optax.apply_updates(state.q_params, updates)
    return TrainingState(q_params, state.target_params, opt_state), loss

if __name__ == "__main__":
    args = parse_args()

    key = jax.random.PRNGKey(args.seed)

    env = create_env(args)
    obs = env.reset()[0]

    q_network = hk.without_apply_rng(hk.transform(lambda obs: Network(env.action_space.n)(obs)))
    target_network = hk.without_apply_rng(hk.transform(lambda obs: Network(env.action_space.n)(obs)))
    optimizer = optax.adam(args.learning_rate)

    initial_q_params = q_network.init(key, obs)
    initial_target_params = initial_q_params
    initial_opt_state = optimizer.init(initial_q_params)
    state = TrainingState(initial_q_params, initial_target_params, initial_opt_state)

    experience_replay = ExperienceReplay(int(args.replay_memory_size), env)
    done = False
    loss_list = []

    for step in range(args.total_steps):
        if done:
            obs = env.reset()[0]
        rand_num = np.random.rand()
        if rand_num < args.epsilon:     
            action = np.asarray(env.action_space.sample())    
        else:
            action = np.asarray(policy(state, jax.device_put(obs)))

        next_obs, reward, done, info, _ = env.step(action)
        
        experience_replay.add(obs, next_obs, action, reward, done)

        obs = next_obs

        if step > args.training_start and step % args.q_network_update_frequency == 0:
            batch = experience_replay.sample(args.minibatch_size)
            state, training_loss = update(state, batch)
            loss_list.append(training_loss)

            if step % args.target_network_update_frequency == 0:
                state = state._replace(target_params=state.q_params)
        Gre
    env.close()

    plt.figure()
    plt.plot(loss_list)
    plt.show()
