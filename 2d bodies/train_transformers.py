from soccer_env import SoccerEnv, ReplayBuffer
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

GRID_WIDTH = 9
GRID_HEIGHT = 7

def random_policy(env):
    actions = {}
    for player in ['A1', 'A2', 'B1', 'B2']:
        move_directions = ['MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN', 'PICK']
        shoot_directions = ['SHOOT_LEFT', 'SHOOT_RIGHT', 'SHOOT_UP', 'SHOOT_DOWN']
        available_actions = move_directions + (shoot_directions if player == env.ball_possession else [])
        chosen_action = random.choice(available_actions)
        actions[player] = chosen_action
    return actions

def action_to_index(action):
    action_mapping = {
        'MOVE_LEFT': 0, 
        'MOVE_RIGHT': 1, 
        'MOVE_UP': 2, 
        'MOVE_DOWN': 3, 
        'PICK': 4, 
        'SHOOT_LEFT': 5, 
        'SHOOT_RIGHT': 6, 
        'SHOOT_UP': 7, 
        'SHOOT_DOWN': 8
    }
    return action_mapping[action]

class BallTransformerSeq(nn.Module):
    def __init__(self, embedding_dim=64, nhead=2, num_layers=2):
        super(BallTransformerSeq, self).__init__()
        self.embedding = nn.Embedding(64, embedding_dim)  # 63 grid positions + ball possession
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim * 6, 63)  # 63 possible positions on the board (7x9)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Swap batch and sequence dimensions
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Swap back to [batch, seq, feature]
        x = x.reshape(x.size(0), -1)  # Flatten all features
        x = self.fc(x)
        return x

class PlayersTransformerSeq(nn.Module):
    def __init__(self, embedding_dim=64, nhead=2, num_layers=2):
        super(PlayersTransformerSeq, self).__init__()
        self.embedding = nn.Embedding(64, embedding_dim)  # 63 grid positions + ball possession
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim * 6, 63)  # 63 possible positions on the board (7x9)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Swap batch and sequence dimensions
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Swap back to [batch, seq, feature]
        x = x.reshape(x.size(0), -1)  # Flatten all features
        x = self.fc(x)
        return x

def preprocess_experience_ball(experiences):
    states_list = []
    for exp in experiences:
        state = exp[0]
        players_flat_positions = []
        for player, pos in state.items():
            players_flat_positions.append(pos[0] * GRID_HEIGHT + pos[1])
        
        ball_x = min(int(exp[1][0] / 40), GRID_WIDTH - 1)  
        ball_y = min(int(exp[1][1] / 40), GRID_HEIGHT - 1)
        ball_flat_position = ball_x * GRID_HEIGHT + ball_y
        ball_possession = 0 if exp[2] is None else 1
        
        state_representation = players_flat_positions + [ball_flat_position, ball_possession]
        states_list.append(state_representation)
    return torch.tensor(states_list, dtype=torch.long)

def preprocess_experience_players(experiences):
    states_list = []
    for exp in experiences:
        state = exp[0]
        players_flat_positions = []
        for player, pos in state.items():
            players_flat_positions.append(pos[0] * GRID_HEIGHT + pos[1])
        
        ball_x = min(int(exp[1][0] / 40), GRID_WIDTH - 1)  
        ball_y = min(int(exp[1][1] / 40), GRID_HEIGHT - 1)
        ball_flat_position = ball_x * GRID_HEIGHT + ball_y
        ball_possession = 0 if exp[2] is None else 1
        
        state_representation = players_flat_positions + [ball_flat_position, ball_possession]
        states_list.append(state_representation)
    return torch.tensor(states_list, dtype=torch.long)

def create_target_ball_heatmap(experiences):
    heatmaps = []
    for exp in experiences:
        heatmap = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        ball_x = min(int(exp[1][0] / 40), GRID_WIDTH - 1)  
        ball_y = min(int(exp[1][1] / 40), GRID_HEIGHT - 1)
        reward = exp[4]
        heatmap[ball_x, ball_y] = reward
        heatmaps.append(heatmap.flatten())
    return torch.tensor(heatmaps, dtype=torch.float)

def create_target_players_heatmap(experiences):
    heatmaps = []
    for _ in experiences:
        heatmap = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        heatmaps.append(heatmap.flatten())
    return torch.tensor(heatmaps, dtype=torch.float)

# Instantiate the models and replay buffer
replay_buffer = ReplayBuffer(10000)
env = SoccerEnv()
env.play_game(random_policy, replay_buffer, num_games=5)

model_ball = BallTransformerSeq()
optimizer_ball = torch.optim.Adam(model_ball.parameters(), lr=0.001)

model_players = PlayersTransformerSeq()
optimizer_players = torch.optim.Adam(model_players.parameters(), lr=0.001)

num_epochs = 20
batch_size = 32

ball_losses = []
players_losses = []

for epoch in tqdm(range(num_epochs), desc="Training"):
    batch = random.sample(replay_buffer.buffer, batch_size)
    ball_states = preprocess_experience_ball(batch)
    players_states = preprocess_experience_players(batch)
    ball_targets = create_target_ball_heatmap(batch)
    players_targets = create_target_players_heatmap(batch)

    ball_logits = model_ball(ball_states)
    players_logits = model_players(players_states)

    ball_loss = nn.MSELoss()(ball_logits, ball_targets)
    players_loss = nn.MSELoss()(players_logits, players_targets)

    optimizer_ball.zero_grad()
    ball_loss.backward()
    optimizer_ball.step()

    optimizer_players.zero_grad()
    players_loss.backward()
    optimizer_players.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Ball Loss: {ball_loss.item():.6f}, Players Loss: {players_loss.item():.6f}")

    ball_losses.append(ball_loss.item())
    players_losses.append(players_loss.item())


plt.plot(ball_losses, label='Ball Transformer Loss')
plt.plot(players_losses, label='Players Transformer Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

torch.save(model_ball.state_dict(), 'model.pth')
torch.save(model_players.state_dict(), 'model.pth')


def preprocess_experience(experiences):
    states_list = []
    for exp in experiences:
        state = exp[0]
        players_flat_positions = []
        for player, pos in state.items():
            players_flat_positions.append(pos[0] * GRID_HEIGHT + pos[1])

        ball_x = min(int(exp[1][0] / 40), GRID_WIDTH - 1)
        ball_y = min(int(exp[1][1] / 40), GRID_HEIGHT - 1)
        ball_flat_position = ball_x * GRID_HEIGHT + ball_y
        ball_possession = 0 if exp[2] is None else 1

        state_representation = players_flat_positions + [ball_flat_position, ball_possession]
        states_list.append(state_representation)
    return torch.tensor(states_list, dtype=torch.long)

# Update the plot_board_state function to accept ax as a parameter
def plot_board_state(state, heatmap, ball_pos, ax):
    ax.imshow(heatmap, cmap='viridis_r', interpolation='nearest')
    
    # Plot players
    player_colors = {
        'A1': 'blue', 'A2': 'skyblue',
        'B1': 'green', 'B2': 'lightgreen'
    }
    
    for player in ['A1', 'A2', 'B1', 'B2']:
        x, y = state[player][0], state[player][1]
        ax.scatter(x, y, color=player_colors[player], s=300, label=player, edgecolors='black')
    
    # Check and plot ball
    ball_x, ball_y = ball_pos
    ax.scatter(ball_x, ball_y, color='red', s=100, label='Ball', edgecolors='black', marker='s')
    
    # Legend off the field
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Grid aesthetics
    ax.set_xticks(np.arange(-.5, GRID_WIDTH, 1))
    ax.set_yticks(np.arange(-.5, GRID_HEIGHT, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='grey', linestyle='-', linewidth=1)
    
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT - 0.5)
    
    plt.colorbar(ax.get_images()[0], ax=ax)

# Visualization for both Ball Transformer and Players Transformer
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Add a title or text to indicate the goal post positions
ax1.text(0.5, -0.1, "Left: Player 1's goal posts | Right: Player 2's goal posts", transform=ax1.transAxes, fontsize=12,
         horizontalalignment='center')

# Visualization for the Ball Transformer
example_experience = random.choice(replay_buffer.buffer)
example_state = preprocess_experience([example_experience])

# Assuming 'ball_possession' information is at index 2 and 'ball_pos' information is at index 1 in example_experience
ball_possession = example_experience[2]
ball_pos = (example_experience[1][0] // 40, example_experience[1][1] // 40)

# Create the state dictionary
state_dict = example_experience[0].copy()  # Make a copy of the state dictionary
if ball_possession is not None:
    state_dict['ball_possession'] = ball_possession
if ball_pos is not None:
    state_dict['ball_pos'] = ball_pos

with torch.no_grad():
    heatmap_ball = model_ball(example_state).squeeze().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)

# Plot the state on the heatmap for the Ball Transformer
plot_board_state(state_dict, heatmap_ball, ball_pos, ax=ax1)
ax1.set_title("Ball Transformer Heatmap")

# Visualization for the Players Transformer
example_experience = random.choice(replay_buffer.buffer)
example_state = preprocess_experience([example_experience])

# Assuming 'ball_possession' information is at index 2 and 'ball_pos' information is at index 1 in example_experience
ball_possession = example_experience[2]
ball_pos = (example_experience[1][0] // 40, example_experience[1][1] // 40)

# Create the state dictionary
state_dict = example_experience[0].copy()  # Make a copy of the state dictionary
if ball_possession is not None:
    state_dict['ball_possession'] = ball_possession
if ball_pos is not None:
    state_dict['ball_pos'] = ball_pos

with torch.no_grad():
    heatmap_players = model_players(example_state).squeeze().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)

# Plot the state on the heatmap for the Players Transformer
plot_board_state(state_dict, heatmap_players, ball_pos, ax=ax2)
ax2.set_title("Players Transformer Heatmap")

# Show both heatmaps side by side with legend in the middle
plt.tight_layout()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.show()
