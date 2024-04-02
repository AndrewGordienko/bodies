import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy 
from torch.distributions.categorical import Categorical
import time
from torch.cuda.amp import GradScaler, autocast

#env = gym.make('CartPole-v0')
#env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('BipedalWalker-v3')
# env = gym.make('LunarLander-v2', render_mode="human")
# env = gym.make('LunarLanderContinuous-v2', render_mode="human")
env = gym.make('BipedalWalker-v3', render_mode="human")

EPISODES = 501
MEM_SIZE = 1000000
BATCH_SIZE = 2056
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
LEARNING_RATE = 0.0003
FC1_DIMS = 1024
FC2_DIMS = 512
ENTROPY_BETA = 0.02  # This is the hyperparameter that controls the strength of the entropy regularization. You might need to tune it.
DEVICE = torch.device("mps")

best_reward = float("-inf")
average_reward = 0
episode_number = []
average_reward_number = []

class actor_network(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.shape
        self.std = 0.5

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, *self.action_space)

        self.log_std = nn.Parameter(torch.ones(1, *self.action_space) * self.std)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.log_std_min = -20  # min bound for log standard deviation
        self.log_std_max = 2    # max bound for log standard deviation

        self.to(DEVICE)


    def net(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))   # Use tanh for the last layer

        return x
    
    def forward(self, x):
        mu = self.net(x)
        # Clipping the log std deviation between predefined min and max values
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        policy_dist = torch.distributions.Normal(mu, std)
        return policy_dist
    
class critic_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = 10
        self.fc1 = nn.Linear(self.input_shape, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x


class PPOMemory:
    def __init__(self, mem_size, state_shape, action_shape, batch_size):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_counter = 0

        # Preallocate memory
        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *action_shape), dtype=np.float32)
        self.prob_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.val_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=bool)

    def store_memory(self, state, action, probs, vals, reward, done):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.prob_memory[index] = probs
        self.val_memory[index] = vals
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.mem_counter += 1

    def generate_batches(self):
        # Minimize memory size to the actual amount of stored experiences
        actual_size = min(self.mem_counter, self.mem_size)

        # Randomly sample indices for the batches
        indices = np.random.choice(actual_size, self.batch_size, replace=False)

        # Extract batches based on sampled indices
        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        probs = self.prob_memory[indices]
        vals = self.val_memory[indices]
        rewards = self.reward_memory[indices]
        dones = self.done_memory[indices]

        return states, actions, probs, vals, rewards, dones, indices

    def clear_memory(self):
        self.mem_counter = 0


class Agent:
    def __init__(self, n_actions, input_dims, alpha=0.0003):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 10
        self.gae_lambda = 0.95
        self.batch_size = BATCH_SIZE
        self.device = DEVICE

        self.actor = actor_network()
        self.critic = critic_network()
        self.encoder = MLP_encoder()
        
        # Ensure action_shape is a tuple
        action_shape = (env.action_space.shape[0],) # This converts the Box shape to a tuple

        self.memory = PPOMemory(mem_size=MEM_SIZE, 
                                state_shape=input_dims, 
                                action_shape=action_shape,  # Use the tuple here
                                batch_size=BATCH_SIZE)
        self.scaler = GradScaler()

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device, non_blocking=True)
        with autocast():
            dist = self.actor(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            value = self.critic(state)

        action = action.cpu().numpy()[0]
        return action, log_prob.item(), value.item()
    
    @staticmethod
    def calculate_tensor_size_in_bytes(*tensors):
        total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        return total_bytes

    @staticmethod
    def bytes_to_gb(bytes):
        return bytes / (1024 ** 3)  # From bytes to GB

    def learn(self):
        start_time = time.time()
        if self.memory.mem_counter < self.batch_size:
            print("Not enough samples to learn")
            return  # Not enough samples to learn

        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, _ = self.memory.generate_batches()

        advantages = np.zeros(len(reward_arr), dtype=np.float32)
        for t in reversed(range(len(reward_arr) - 1)):
            delta = reward_arr[t] + self.gamma * (vals_arr[t + 1] * (1 - int(dones_arr[t]))) - vals_arr[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (advantages[t + 1] if t < len(reward_arr) - 1 else 0)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        state_arr = torch.tensor(state_arr, dtype=torch.float).to(self.device)
        action_arr = torch.tensor(action_arr).to(self.device)
        old_prob_arr = torch.tensor(old_prob_arr, dtype=torch.float).to(self.device)
        vals_arr = torch.tensor(vals_arr, dtype=torch.float32).to(self.device)

        for _ in range(self.n_epochs):
            permutation = torch.randperm(state_arr.size(0))
            for i in range(0, state_arr.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                states, actions, old_probs, advantages_batch, vals = state_arr[indices], action_arr[indices], old_prob_arr[indices], advantages[indices], vals_arr[indices]

                bytes_used = self.calculate_tensor_size_in_bytes(states, actions, old_probs, advantages_batch, vals)
                size_in_gb = self.bytes_to_gb(bytes_used)
                
                # Debug prints
                print(f"Total bytes for batch: {bytes_used}")
                print(f"Shapes - states: {states.shape}, actions: {actions.shape}, old_probs: {old_probs.shape}, advantages_batch: {advantages_batch.shape}, vals: {vals.shape}")
                print(f"Batch size: {size_in_gb:.6f} GB")

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()
                new_probs = dist.log_prob(actions).sum(axis=-1)
                prob_ratios = torch.exp(new_probs - old_probs)
                surr1 = prob_ratios * advantages_batch
                surr2 = torch.clamp(prob_ratios, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                returns = advantages_batch + vals
                critic_loss = F.mse_loss(returns, critic_value)
                total_loss = actor_loss + 0.5 * critic_loss - ENTROPY_BETA * dist.entropy().mean()

                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        end_time = time.time()
        print(f"learn function duration: {end_time - start_time:.2f} seconds")




    def calculate_advantages(self, rewards, values, dones):
        """
        Compute advantages in a vectorized manner.
        """
        advantages = torch.zeros_like(rewards).to(DEVICE)
        next_value = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Value of next state is 0 because it's terminal
            else:
                next_value = values[t + 1]
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + GAMMA * self.gae_lambda * (1 - dones[t]) * (advantages[t + 1] if t < len(rewards) - 1 else 0)

        return advantages



agent = Agent(n_actions=env.action_space, input_dims=env.observation_space.shape)
torch.save(agent.encoder.state_dict(), 'agent_encoder.pth')
max_steps = 0

for i in range(1, EPISODES):
    observation, info = env.reset()
    score = 0
    done = False
    step = 0
    total_training_time = 0
    start_time = time.time()

    while not done:
        env.render()
        step += 1

        observation = np.concatenate((
            observation[:len(observation)-10], 
            agent.encoder(torch.Tensor(observation[len(observation)-10:])).detach().numpy()
        ))

        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, truncated, info = env.step(action)
        score += reward
        agent.remember(observation, action, prob, val, reward, done)

        
        if step % 20 == 0:
            agent.learn()
        

        observation = observation_

        
        if done:
            torch.save(agent.actor.state_dict(), 'agent_actor.pth')
            
            if score > best_reward:
                best_reward = score
                # torch.save(agent.actor.state_dict(), 'agent_actor.pth')
            
            """
            if step > max_steps:
                torch.save(agent.actor.state_dict(), 'agent_actor.pth')
                max_steps = step
            """

            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {}".format(i, average_reward/i, best_reward, score))

            end_time = time.time()
            episode_duration = end_time - start_time
            total_training_time += episode_duration
            print(f"Total training time: {total_training_time:.2f}s Steps {step}")

            break
            
        episode_number.append(i)
        average_reward_number.append(average_reward/i)
    
    """
    agent.learn()
    agent.memory = PPOMemory(BATCH_SIZE)
    """

plt.plot(episode_number, average_reward_number)
plt.show()
