import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import gym
import random

from experience_replay import ExperienceReplay

class Network(nn.Module):

    @nn.compact
    def __call__(self, x):

        x = nn.Dense(features=4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)

        return x


class DQN:

    def __init__(self, env):

        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.experience_replay = ExperienceReplay(max_len=1000)
        self.epsilon = 0.05
        self.q_net = Network()
        seed = jax.random.PRNGKey(0)
        self.params_q = self.q_net.init(seed, jax.random.normal(key=seed, shape=(1, self.state_size)))
        self.target_net = Network()
        self.params_target = self.target_net.init(seed, jax.random.normal(key=seed, shape=(1, self.state_size)))
        self.optimizer = optax.adam(1e-3)
        self.optimizer_state = self.optimizer.init(self.params_q)


    
    def policy(self, state):

        rand_num = np.random.rand()
        if rand_num <= self.epsilon:
            return 0 if rand_num < 0.025 else 1
        q_values = self.q_net.apply(self.params_q, state)
        action = jnp.argmax(q_values).item()
        return action

    def compute_loss(self, batch):

        states, actions, rewards, next_states, dones = zip(*batch)
        states, rewards, next_states, dones = jnp.array(states), jnp.array(rewards), jnp.array(next_states), jnp.array(dones)
        q_values = self.target_net.apply(self.params_target, next_states)
        target_q_values = rewards + (1 - dones) * np.max(q_values)
        loss = jnp.mean(jnp.square(target_q_values - np.max(q_values)))
        return loss

    def train(self, batch_size=32, epochs=1):

        # Populate experience replay
        for _ in range(1):
            state, _ = self.env.reset()
            print(state)
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ , __ = self.env.step(action)
                self.experience_replay.add((state, action, reward, next_state, done))
                state = next_state

        # Train
        for epoch in range(epochs):
            for _ in range(1):
                state, _ = self.env.reset()
                done = False
                while not done:
                    action = self.policy(state)
                    next_state, reward, done, _, __ = self.env.step(action)
                    self.experience_replay.add((state, action, reward, next_state, done))
                    state = next_state
            batch = self.experience_replay.sample(batch_size)
            loss, gradients = jax.value_and_grad(self.compute_loss, allow_int=True)(batch)
            updates, self.optimizer_state = self.optimizer.update(gradients, self.optimizer_state)
            self.params_q = optax.apply_updates(self.params_q, updates)
            if epoch % 10 == 0:
                self.params_target = self.params_q 
        
        self.env.close()      

if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    dqn = DQN(env)
    dqn.train()