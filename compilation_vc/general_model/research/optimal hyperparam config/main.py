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

#env = gym.make('CartPole-v0')
#env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('BipedalWalker-v3')
# env = gym.make('LunarLander-v2', render_mode="human")
#env = gym.make('LunarLanderContinuous-v2', render_mode="human")
# env = gym.make('BipedalWalker-v3', render_mode="human")

from my_biped import BipedalWalker
env = BipedalWalker()

EPISODES = 20
MEM_SIZE = 1000000
BATCH_SIZE = 5
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
LEARNING_RATE = 0.0003
FC1_DIMS = 1024
FC2_DIMS = 512
FC3_DIMS = 256
ENTROPY_BETA = 0.05  # This is the hyperparameter that controls the strength of the entropy regularization. You might need to tune it.
DEVICE = torch.device("cpu")

best_reward = float("-inf")
average_reward = 0
episode_number = []
average_reward_number = []
all_metrics = {}

hyperparameters = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [5, 10, 15],
    'gae_lambda': [0.9, 0.95],
    'entropy_beta': [0.01, 0.05, 0.1]
}

class actor_network(nn.Module):
    def __init__(self, alpha, entropy_beta):
        super().__init__()

        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.shape
        self.std = 0.5

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, FC3_DIMS)
        self.fc4 = nn.Linear(FC3_DIMS, *self.action_space)

        self.log_std = nn.Parameter(torch.ones(1, *self.action_space) * self.std)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.entropy_beta = entropy_beta  # Store entropy_beta for use in calculations

        self.log_std_min = -20  # min bound for log standard deviation
        self.log_std_max = 2    # max bound for log standard deviation


    def net(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))   # Use tanh for the last layer

        return x
    
    def forward(self, x):
        mu = self.net(x)
        # Clipping the log std deviation between predefined min and max values
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mu)
        policy_dist = torch.distributions.Normal(mu, std)
        return policy_dist
    
class critic_network(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.input_shape = env.observation_space.shape

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, FC3_DIMS)
        self.fc4 = nn.Linear(FC3_DIMS, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.to(DEVICE)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Agent:
    def __init__(self, n_actions, input_dims, alpha, batch_size, gae_lambda, entropy_beta):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 10
        self.gae_lambda = gae_lambda  # Use the passed gae_lambda

        # Modify the actor and critic networks to accept learning rate as a parameter
        self.actor = actor_network(alpha=alpha, entropy_beta=entropy_beta)
        self.critic = critic_network(alpha=alpha)
        self.encoder = MLP_encoder()
        self.memory = PPOMemory(batch_size=batch_size)  # Use the passed batch_size

        # Metrics
        self.actor_losses = []
        self.critic_losses = []
        self.rewards_history = []
        self.best_rewards = []
        self.steps_per_episode = []
        self.average_rewards = []
        self.adjusted_scores = []  # Stores individual adjusted scores
        self.average_adjusted_scores = []  # Stores the average of adjusted scores

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(DEVICE)
        
        # directly assign the Normal distribution object to dist
        dist = self.actor(state)
        
        mu = dist.mean
        sigma = dist.stddev
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic.forward(state)
        
        action = action.cpu().numpy()[0]
        log_prob = log_prob.item()
        value = torch.squeeze(value).item()

        return action, log_prob, value


    def learn(self):
        for i in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            
            advantage = torch.tensor(advantage).to(DEVICE)
            values = torch.tensor(values).to(DEVICE)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(DEVICE)
                old_probs = torch.tensor(old_prob_arr[batch]).to(DEVICE)
                actions = torch.tensor(action_arr[batch]).to(DEVICE)

                dist = self.actor(states)  # Get the policy distribution
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probabilities = dist.log_prob(actions).sum(axis=-1)

                probability_ratio = new_probabilities.exp() / old_probs.exp()

                weighted_probabilities = advantage[batch] * probability_ratio
                weighted_clipped_probabilities = torch.clamp(probability_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                
                entropy = dist.entropy().mean()
                actor_loss = -torch.min(weighted_probabilities, weighted_clipped_probabilities).mean() - ENTROPY_BETA * entropy

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
            
        
        self.memory.clear_memory()   

# Setup for real-time plotting
plt.ion()
fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # Create a 3x3 subplot grid
fig.suptitle('Training Metrics')

# Adjust the lines dictionary to include all 7 metrics
lines = {
    'Actor Loss': axs[0, 0].plot([], [])[0],
    'Critic Loss': axs[0, 1].plot([], [])[0],
    'Rewards': axs[0, 2].plot([], [])[0],
    'Adjusted Average Reward': axs[1, 0].plot([], [])[0],
    'Step Count': axs[1, 1].plot([], [])[0],
    'Average Reward': axs[1, 2].plot([], [])[0],
    'Best Reward': axs[2, 0].plot([], [])[0],
    # Adding convergence plots for Actor and Critic Losses
    'Convergence (Actor Loss)': axs[2, 1].plot([], [], label='Actor Loss')[0],
    'Convergence (Critic Loss)': axs[2, 1].plot([], [], label='Critic Loss')[0],
}
axs[2, 1].set_title('Convergence')
axs[2, 1].legend()

for title, ax in zip(lines.keys(), axs.flat):
    ax.set(title=title)
    ax.label_outer()

num_metrics = 7

# Initialize a figure for cumulative summary across configurations
cumulative_fig, cumulative_axs = plt.subplots(num_metrics, 1, figsize=(10, 15))
cumulative_fig.subplots_adjust(hspace=0.4, wspace=0.4)
cumulative_fig.suptitle('Cumulative Summary Across Configurations')

# Individual figure for each configuration
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle('Training Metrics')

legend_fig = None
legend_ax = None

def update_plots(metrics_dictionary):
    global legend_fig, legend_ax  # Use global variables to track the legend window

    # Titles for individual plots
    metric_titles = ['Actor Loss', 'Critic Loss', 'Rewards', 'Step Count', 'Average Reward', 'Best Reward', 'Adjusted Average Reward']

    # Clear previous plots
    for ax in axs.flat:
        ax.clear()
    
    # Reset handles for each update to avoid duplicate entries
    legend_handles = {}
    
    # Loop through each configuration to plot and summarize metrics
    for config_key, metrics in enumerate(metrics_dictionary.items()):
        for i, metric_key in enumerate(['actor_losses', 'critic_losses', 'rewards_history', 'steps_per_episode', 'average_rewards', 'best_rewards', 'average_adjusted_scores']):
            if metric_key in metrics[1]:  # Access the metrics dictionary correctly
                line, = axs.flatten()[i].plot(np.arange(len(metrics[1][metric_key])), metrics[1][metric_key], label=metrics[0])
                # Adding titles to individual plots
                if i < len(metric_titles):
                    axs.flatten()[i].set_title(metric_titles[i])
                else:
                    axs.flatten()[i].set_title('')
                # Store only the first handle for each configuration
                if metrics[0] not in legend_handles:
                    legend_handles[metrics[0]] = line

    # Titles and labels for cumulative summary plots
    for i, metric_title in enumerate(metric_titles):
        cumulative_axs[i].set_title(metric_title)
        cumulative_axs[i].set_xlabel('Configuration')
        cumulative_axs[i].set_ylabel(metric_title)

    # Loop through each configuration to plot and summarize metrics in cumulative plots
    for config_key, metrics in metrics_dictionary.items():
        for i, metric_key in enumerate(['actor_losses', 'critic_losses', 'rewards_history', 'steps_per_episode', 'average_rewards', 'best_rewards', 'average_adjusted_scores']):
            if metric_key in metrics:
                average_metric_value = np.mean(metrics[metric_key])
                cumulative_axs[i].bar(config_key, average_metric_value, label=config_key)

    # Check if the legend figure exists. If not, create it; otherwise, clear it for updating
    if legend_fig is None or legend_ax is None:
        legend_fig = plt.figure(figsize=(5, 3))
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis('off')
    else:
        legend_ax.clear()

    # Update the legend with the current configurations
    legend_ax.legend(handles=list(legend_handles.values()), labels=list(legend_handles.keys()), 
                 title="Configurations", loc='upper left', fontsize='small')  # Using a predefined string value
    legend_fig.canvas.draw_idle()  # Redraw the legend figure to update it

    plt.tight_layout()
    plt.show()



# def update_plots(metrics_dictionary):
#     for config_key, metrics in metrics_dictionary.items():
#         # Create a new figure for each configuration
#         fig, axs = plt.subplots(3, 3, figsize=(15, 15))
#         fig.suptitle(f'Training Metrics for {config_key}')
        
#         # Update data for each metric in the current configuration
#         print("---")
#         print(metrics['actor_losses'])
#         print(metrics['critic_losses'])
#         print(metrics['rewards_history'])
#         print(metrics['steps_per_episode'])
#         print(metrics['average_rewards'])
#         print(metrics['best_rewards'])
#         print(metrics['average_adjusted_scores'])
#         print("down here")

#         axs[0, 0].plot(np.arange(len(metrics['actor_losses'])), metrics['actor_losses'], label='Actor Loss')
#         axs[0, 1].plot(np.arange(len(metrics['critic_losses'])), metrics['critic_losses'], label='Critic Loss')
#         axs[0, 2].plot(np.arange(len(metrics['rewards_history'])), metrics['rewards_history'], label='Rewards')
#         axs[1, 0].plot(np.arange(len(metrics['steps_per_episode'])), metrics['steps_per_episode'], label='Step Count')
#         axs[1, 1].plot(np.arange(len(metrics['average_rewards'])), metrics['average_rewards'], label='Average Reward')
#         axs[1, 2].plot(np.arange(len(metrics['best_rewards'])), metrics['best_rewards'], label='Best Reward')
#         axs[2, 0].plot(np.arange(len(metrics['average_adjusted_scores'])), metrics['average_adjusted_scores'], label='Adjusted Average Reward')
        
#         # Set titles and labels for each subplot
#         axs[0, 0].set_title('Actor Loss')
#         axs[0, 1].set_title('Critic Loss')
#         axs[0, 2].set_title('Rewards')
#         axs[1, 0].set_title('Step Count')
#         axs[1, 1].set_title('Average Reward')
#         axs[1, 2].set_title('Best Reward')
#         axs[2, 0].set_title('Adjusted Average Reward')
        
#     # Adjust layout
#     for ax in axs.flat:
#         ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
#         ax.legend()

#     plt.tight_layout()
#     plt.show()


# def update_plots():
#     # Update data for each metric
#     print("--")
#     print(metrics_dictionary)

#     lines['Actor Loss'].set_data(np.arange(len(agent.actor_losses)), agent.actor_losses)
#     lines['Critic Loss'].set_data(np.arange(len(agent.critic_losses)), agent.critic_losses)
#     lines['Rewards'].set_data(np.arange(len(agent.rewards_history)), agent.rewards_history)
#     lines['Step Count'].set_data(np.arange(len(agent.steps_per_episode)), agent.steps_per_episode)
#     lines['Average Reward'].set_data(np.arange(len(agent.average_rewards)), agent.average_rewards)
#     lines['Best Reward'].set_data(np.arange(len(agent.best_rewards)), agent.best_rewards)
#     lines['Adjusted Average Reward'].set_data(np.arange(len(agent.average_adjusted_scores)), agent.average_adjusted_scores)
    
#     # Update the convergence plot separately
#     lines['Convergence (Actor Loss)'].set_data(np.arange(len(agent.actor_losses)), agent.actor_losses)
#     lines['Convergence (Critic Loss)'].set_data(np.arange(len(agent.critic_losses)), agent.critic_losses)

#     # Adjust layout and explicitly enable tick labels
#     fig.tight_layout(pad=3.0)  # Adjust padding between and around subplots
#     for ax in axs.flat:
#         ax.relim()  # Recalculate limits
#         ax.autoscale_view()  # Rescale view based on limits
#         ax.set_xlabel('Episode')  # Set the common X-axis label
#         ax.set_ylabel('Value')   # Placeholder Y-axis label, adjust as necessary
        
#         # Enable tick labels explicitly
#         ax.xaxis.set_tick_params(labelbottom=True)
#         ax.yaxis.set_tick_params(labelleft=True)

#     # Adjust subplot spacing if needed, especially for tight_layout
#     fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust if the labels still overlap

#     # Force redraw of the canvas
#     fig.canvas.draw()
#     fig.canvas.flush_events()

average_rewards = []
episodes = []

def update_plots_with_best_fit():
    # Update data points
    update_plots()  # Updates the raw data points on the plots

    # Iterate over each plot to calculate and update lines of best fit
    for key, line in lines.items():
        ax = line.axes
        data_x = line.get_xdata()
        data_y = line.get_ydata()
        
        # Ensure there are enough points to calculate a line of best fit
        if len(data_x) > 1 and len(data_y) > 1:
            # Calculate coefficients of the linear regression line (slope and intercept)
            slope, intercept = np.polyfit(data_x, data_y, 1)
            # Generate y values based on the line of best fit
            best_fit_y = slope * data_x + intercept
            
            # Check if best_fit line exists; create or update it
            if 'best_fit' not in ax.__dict__:
                ax.best_fit, = ax.plot(data_x, best_fit_y, label=f'{key} Best Fit', linestyle='--')
            else:
                ax.best_fit.set_data(data_x, best_fit_y)
                
            # Now, only call legend() if there are labeled artists
            if any([artist.get_label() for artist in ax.get_lines()]):
                ax.legend()

    # Redraw the figure to show updates
    fig.canvas.draw()
    fig.canvas.flush_events()

best_configuration = {}
best_reward = float('-inf')
metrics_dictionary = {}

# Iterate over all combinations of hyperparameters
for lr in hyperparameters['learning_rate']:
    for bs in hyperparameters['batch_size']:
        for gae in hyperparameters['gae_lambda']:
            for entropy in hyperparameters['entropy_beta']:
                # Initialize your agent with the current set of hyperparameters
                agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape,
                              alpha=lr, batch_size=bs, gae_lambda=gae, entropy_beta=entropy)
                
                config_key = f"LR={lr}_BS={bs}_GAE={gae}_ENT={entropy}"
                print(f"hyperparameters. learning rate {lr}, batch size {bs}, gae {gae}, entropy {entropy}")
                max_steps = 0


                for i in range(1, EPISODES):
                    observation, info = env.reset()
                    score = 0
                    done = False
                    step = 0

                    while not done:
                        env.render()
                        step += 1

                        observation = np.concatenate((
                            observation[:len(observation)-10], 
                            agent.encoder(torch.Tensor(observation[len(observation)-10:])).detach().numpy()
                        ))

                        action, prob, val = agent.choose_action(observation)
                        observation_, reward, done, info = env.step(action)
                        score += reward
                        agent.remember(observation, action, prob, val, reward, done)

                        
                        if step % 20 == 0:
                            agent.learn()
                        

                        observation = observation_

                        
                        if done:
                            torch.save(agent.actor.state_dict(), 'agent_actor.pth')
                            
                            average_reward += score 

                            agent.rewards_history.append(score)
                            average_reward = sum(agent.rewards_history) / len(agent.rewards_history)

                            adjusted_score = score / step  # Calculate adjusted score
                            agent.adjusted_scores.append(adjusted_score)  # Store adjusted score
                            average_adjusted_score = sum(agent.adjusted_scores) / len(agent.adjusted_scores)    
                            agent.average_adjusted_scores.append(average_adjusted_score)  # Assuming you have this list to store averages


                            agent.best_rewards.append(best_reward)
                            agent.average_rewards.append(average_reward)
                            agent.steps_per_episode.append(step)

                            # Update plot

                            print("Episode {} Average Reward {} Best Reward {} Last Reward {} Steps {}".format(i, average_reward/i, best_reward, score, step))
                            if score > best_reward:
                                best_reward = score
                                best_configuration = {
                                    'learning_rate': lr,
                                    'batch_size': bs,
                                    'gae_lambda': gae,
                                    'entropy_beta': entropy
                                }
                    

                            break
                            
                        episode_number.append(i)
                        average_reward_number.append(average_reward/i)
                    
                    
                    """
                    agent.learn()
                    agent.memory = PPOMemory(BATCH_SIZE)
                    """
                
                metrics_dictionary[config_key] = {
                'average_rewards': agent.average_rewards.copy(),
                'best_rewards': agent.best_rewards.copy(),
                'actor_losses': agent.actor_losses.copy(),
                'critic_losses': agent.critic_losses.copy(),
                'steps_per_episode': agent.steps_per_episode.copy(),
                'adjusted_scores': agent.adjusted_scores.copy(),
                'average_adjusted_scores': agent.average_adjusted_scores.copy(),
                'rewards_history': agent.rewards_history.copy()
                }

                # update_plots_with_best_fit()
                print("Updating config with the following average metrics: " + config_key)
                print(f"Actor Losses Mean: {np.mean(agent.actor_losses):.4f}")
                print(f"Critic Losses Mean: {np.mean(agent.critic_losses):.4f}")
                print(f"Last Actor Loss: {agent.actor_losses[-1]:.4f}")
                print(f"Last Critic Loss: {agent.critic_losses[-1]:.4f}")
                print(f"Steps per Episode Mean: {np.mean(agent.steps_per_episode):.4f}")
                print(f"Average Adjusted Scores Mean: {np.mean(agent.average_adjusted_scores):.4f}")
                print(f"Rewards History Mean: {np.mean(agent.rewards_history):.4f}")
                print(f"Best Rewards Mean: {agent.best_rewards[-1]:.4f}")
                print(f"Average Reward: {agent.average_rewards[-1]:.4f}\n")

                with open("data.txt", "a") as f:
                    f.write(f"Updating config with the following average metrics: {config_key}\n")
                    f.write(f"Actor Losses Mean: {np.mean(agent.actor_losses):.4f}\n")
                    f.write(f"Critic Losses Mean: {np.mean(agent.critic_losses):.4f}\n")
                    f.write(f"Last Actor Loss: {agent.actor_losses[-1]:.4f}")
                    f.write(f"Last Critic Loss: {agent.critic_losses[-1]:.4f}")
                    f.write(f"Steps per Episode Mean: {np.mean(agent.steps_per_episode):.4f}\n")
                    f.write(f"Average Adjusted Scores Mean: {agent.average_adjusted_scores[-1]:.4f}\n")
                    f.write(f"Rewards History Mean: {np.mean(agent.rewards_history):.4f}\n")
                    f.write(f"Best Rewards Mean: {agent.best_rewards[-1]:.4f}\n")
                    f.write(f"Average Reward: {agent.average_rewards[-1]:.4f}\n")


                update_plots(metrics_dictionary)

print("the best scoring configuration was")
print(best_configuration)
# Keep the plot open at the end of the loop
plt.ioff()
plt.show()  
