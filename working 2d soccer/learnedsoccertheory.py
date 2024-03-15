import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import math
from changed_soccer_env import SoccerEnv
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)


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

class ValueNet(nn.Module):
    def __init__(self, embedding_dim=64, nhead=2, num_layers=2):
        super(ValueNet, self).__init__()
        self.embedding = nn.Embedding(64, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(embedding_dim * 6, 128)
        self.fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        value = self.tanh(self.fc2(x))
        return value.squeeze(-1)
    
class Supplementary():
    def state_to_matrix(self, state):
        matrix = np.zeros((env.GRID_HEIGHT, env.GRID_WIDTH))
        player_mapping = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4}
        for player, position in state['player_positions'].items():
            matrix[position[1], position[0]] = player_mapping[player]
        matrix[state['ball_pos'][1]][state['ball_pos'][0]] += 10
        return matrix

    def preprocess_board_state_sequence(self, states):
        state_sequence = []
        for state in states:
            players_flat_positions = [pos[0] * env.GRID_HEIGHT + pos[1] for pos in state['player_positions'].values()]
            ball_x = min(int(state['ball_pos'][0] // env.CELL_SIZE), env.GRID_WIDTH - 1)
            ball_y = min(int(state['ball_pos'][1] // env.CELL_SIZE), env.GRID_HEIGHT - 1)
            ball_flat_position = ball_x * env.GRID_HEIGHT + ball_y
            ball_possession = 0 if state['ball_possession'] is None else 1
            state_sequence.append(players_flat_positions + [ball_flat_position, ball_possession])
        return torch.tensor([state_sequence], dtype=torch.long).squeeze(0)

    def get_surrounding_actions(self, player_pos, logits, is_team1=True):
        x, y = player_pos
        surrounding_values = {}

        if x > 0: surrounding_values['MOVE_LEFT'] = logits[x-1, y]
        if x < env.GRID_WIDTH - 1: surrounding_values['MOVE_RIGHT'] = logits[x+1, y]
        if y > 0: surrounding_values['MOVE_UP'] = logits[x, y-1]
        if y < env.GRID_HEIGHT - 1: surrounding_values['MOVE_DOWN'] = logits[x, y+1]

        # Filter for team preferences
        filtered_values = {k: v for k, v in surrounding_values.items() if (is_team1 and v > 0) or (not is_team1 and v < 0)}
        
        # Check if we have at least two moves in filtered_values
        if len(filtered_values) < 2:
            # If not, sort all actions by absolute value and take the two with the highest absolute value
            sorted_actions = sorted(surrounding_values.keys(), key=lambda k: abs(surrounding_values[k]), reverse=True)[:2]
        else:
            # If we have at least two, sort filtered_values by absolute value and take the top 3
            sorted_actions = sorted(filtered_values.keys(), key=lambda k: abs(filtered_values[k]), reverse=True)[:2]
        
        return sorted_actions

    def get_surrounding_actions_ball(self, player_pos, logits, is_team1=True):
        x, y = player_pos
        surrounding_values = {}
        #print(logits.shape)
        
        if x > 1: surrounding_values['SHOOT_LEFT'] = logits[x-1, y]
        if x < env.GRID_WIDTH - 2: surrounding_values['SHOOT_RIGHT'] = logits[x+1, y]
        if y > 1: surrounding_values['SHOOT_UP'] = logits[x, y-1]
        if y < env.GRID_HEIGHT - 2: surrounding_values['SHOOT_DOWN'] = logits[x, y+1]

        # Filter for team preferences
        filtered_values = {k: v for k, v in surrounding_values.items() if (is_team1 and v > 0) or (not is_team1 and v < 0)}
        
        # Check if we have at least two moves in filtered_values
        if len(filtered_values) < 2:
            # If not, sort all actions by absolute value and take the two with the highest absolute value
            sorted_actions = sorted(surrounding_values.keys(), key=lambda k: abs(surrounding_values[k]), reverse=True)[:2]
        else:
            # If we have at least two, sort filtered_values by absolute value and take the top 3
            sorted_actions = sorted(filtered_values.keys(), key=lambda k: abs(filtered_values[k]), reverse=True)[:2]
        
        return sorted_actions
    
    def check_validity(self, action, state, player):
        initial_pos = state['player_positions'][player]
        
        player_pos = state['player_positions'][player]
        ball_pos = state['ball_pos']
        if abs(player_pos[0] - ball_pos[0]) <= 1 and abs(player_pos[1] - ball_pos[1]) <= 1:
            state['ball_possession'] = player
            state['ball_pos'] = deepcopy(state['player_positions'][player])

        # Update the ball's position based on the player's movement if the player possesses the ball
        if state['ball_possession'] == player:
            if action == 'MOVE_LEFT' and initial_pos[0] > 0:
                state['ball_pos'][0] -= 1
            elif action == 'MOVE_RIGHT' and initial_pos[0] < env.GRID_WIDTH - 1:
                state['ball_pos'][0] += 1
            elif action == 'MOVE_UP' and initial_pos[1] > 0:
                state['ball_pos'][1] -= 1
            elif action == 'MOVE_DOWN' and initial_pos[1] < env.GRID_HEIGHT - 1:
                state['ball_pos'][1] += 1
        
        # Update the player's position based on the action
        if action == 'MOVE_LEFT' and state['player_positions'][player][0] > 0:
            state['player_positions'][player][0] -= 1
        elif action == 'MOVE_RIGHT' and state['player_positions'][player][0] < env.GRID_WIDTH - 1:
            state['player_positions'][player][0] += 1
        elif action == 'MOVE_UP' and state['player_positions'][player][1] > 0:
            state['player_positions'][player][1] -= 1
        elif action == 'MOVE_DOWN' and state['player_positions'][player][1] < env.GRID_HEIGHT - 1:
            state['player_positions'][player][1] += 1

        return state

    
    def check_ball_validity(self, action, state):
        if action in ['SHOOT_LEFT', 'SHOOT_RIGHT', 'SHOOT_UP', 'SHOOT_DOWN']:
            if action == 'SHOOT_LEFT' and state['ball_pos'][0] > 1:
                state['ball_pos'][0] -= 2
            elif action == 'SHOOT_RIGHT' and state['ball_pos'][0] < env.GRID_WIDTH - 2:
                state['ball_pos'][0] += 2
            elif action == 'SHOOT_UP' and state['ball_pos'][1] > 1:
                state['ball_pos'][1] -= 2
            elif action == 'SHOOT_DOWN' and state['ball_pos'][1] < env.GRID_HEIGHT - 2:
                state['ball_pos'][1] += 2

            state['ball_possession'] = None
            for player in state['player_positions']:
                if state['player_positions'][player] == [state['ball_pos'][0], state['ball_pos'][1]]:
                    state['ball_possession'] = player

        return state

    def check_goal(self, ball_pos):
        if env.net_top_position <= ball_pos[1] <= env.net_top_position + env.net_height:
            if ball_pos[0] <= 1:  
                return 'B'
            elif ball_pos[0] >= env.SCREEN_WIDTH - 1:
                return 'A'
        return None
    
    def action_to_index(self, action_tuple):
        action_mapping = {'MOVE_LEFT': 0, 'MOVE_RIGHT': 1, 'MOVE_UP': 2, 'MOVE_DOWN': 3, 
                          'SHOOT_LEFT': 4, 'SHOOT_RIGHT': 5, 'SHOOT_UP': 6, 'SHOOT_DOWN': 7}
        index = 0
        for i, action in enumerate(action_tuple):
            index += action_mapping[action] * (8 ** i)
        return index % 63

    def build_state_buffer(self, node):
        state_buffer = [node.state]
        temp_node = node
        while len(state_buffer) < 6 and temp_node.parent:
            state_buffer.insert(0, temp_node.parent.state)
            temp_node = temp_node.parent
        while len(state_buffer) < 6:
            state_buffer.insert(0, state_buffer[0])
        return state_buffer

    def preprocess_states(self, state):
        # Use the supplementary function to handle the preprocessing
        return supplementary.preprocess_board_state_sequence([state] * 6)  # Repeat the state 6 times for the sequence


class Node():
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.children = None
        self.value = 0
        self.visits = 0
        self.action = None  # The move which led to this node

    def create_children(self):
        list_of_children = []
        state_buffer = []

        depth_node = self
        for i in range(6):
            state_buffer.append(depth_node.state)
            if depth_node.parent != None:
                depth_node = depth_node.parent

        state_buffer = [self.state for _ in range(6)] # this needs to be changed so it actually goes back
        preprocessed_state = supplementary.preprocess_board_state_sequence(state_buffer)
        policy_logits = model.forward(preprocessed_state)[0].squeeze().detach().numpy() 
        policy_logits = policy_logits.reshape(env.GRID_WIDTH, env.GRID_HEIGHT)
        ball_policy_logits = ball_model.forward(preprocessed_state)[0].squeeze().detach().numpy() 
        ball_policy_logits = ball_policy_logits.reshape(env.GRID_WIDTH, env.GRID_HEIGHT)

        pos_A1 = self.state["player_positions"]["A1"]
        pos_A2 = self.state["player_positions"]["A2"]
        pos_B1 = self.state["player_positions"]["B1"]
        pos_B2 = self.state["player_positions"]["B2"]

        possession, team = None, None
        
        if self.state["ball_possession"] == "A1": possession, team = pos_A1, True
        if self.state["ball_possession"] == "A2": possession, team = pos_A2, True
        if self.state["ball_possession"] == "B1": possession, team = pos_B1, False
        if self.state["ball_possession"] == "B2": possession, team = pos_B2, False

        top_actions_A1 = supplementary.get_surrounding_actions(pos_A1, policy_logits, True)
        top_actions_A2 = supplementary.get_surrounding_actions(pos_A2, policy_logits, True)
        top_actions_B1 = supplementary.get_surrounding_actions(pos_B1, policy_logits, False)
        top_actions_B2 = supplementary.get_surrounding_actions(pos_B2, policy_logits, False)
        if possession != None: top_ball_moves = supplementary.get_surrounding_actions_ball(possession, policy_logits, team)

        for action_a1 in top_actions_A1:
            for action_a2 in top_actions_A2:
                for action_b1 in top_actions_B1:
                    for action_b2 in top_actions_B2:
                        if possession != None: 
                            for ball_move in top_ball_moves:
                                state = supplementary.check_validity(action_a1, deepcopy(self.state), "A1")
                                state = supplementary.check_validity(action_a2, state, "A2")
                                state = supplementary.check_validity(action_b1, state, "B1")
                                state = supplementary.check_validity(action_b2, state, "B2")
                                state = supplementary.check_ball_validity(ball_move, state)

                                temporary_node = Node(self, deepcopy(state))
                                temporary_node.action = (action_a1, action_a2, action_b1, action_b2, ball_move)
                                list_of_children.append(temporary_node)
                        else:
                            state = supplementary.check_validity(action_a1, deepcopy(self.state), "A1")
                            state = supplementary.check_validity(action_a2, state, "A2")
                            state = supplementary.check_validity(action_b1, state, "B1")
                            state = supplementary.check_validity(action_b2, state, "B2")

                            temporary_node = Node(self, deepcopy(state))
                            temporary_node.action = (action_a1, action_a2, action_b1, action_b2)
                            list_of_children.append(temporary_node)
        
        """
        # Handle ball shooting logic
        policy_logits = ball_model.forward(preprocessed_state)[0].squeeze().detach().numpy() 
        policy_logits = policy_logits.reshape(env.GRID_WIDTH, env.GRID_HEIGHT)
        if possession != None:
            top_ball_moves = supplementary.get_surrounding_actions_ball(possession, policy_logits, team)
            for move in top_ball_moves:
                state = supplementary.check_ball_validity(move, self.state)
                shoot_node = Node(self, state)
                shoot_node.action = move
                list_of_children.append(shoot_node)
        """

        self.children = list_of_children

class MCTS():
    def __init__(self, initial_epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.search_length = 250
        self.depth = 10
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def search(self, state):
        starting_node = Node(None, state)
        starting_node.create_children()

        for i in range(self.search_length):
            new_node = self.selection(starting_node)
            score = self.estimate_value(new_node)
            self.backpropagation(new_node, score)
        
        best_action_value = float("-inf")
        best_child = None
        for child in starting_node.children:
            value = child.value / (child.visits + 1)
            if value > best_action_value:
                best_child = child
                best_action_value = value
        return best_child
    
    def selection(self, node):
        depth = 0
        while supplementary.check_goal(node.state['ball_pos']) == None and depth < self.depth:
            if not node.children or node.visits == 0:
                return node
            else:
                node = self.choose_node(node)
            depth += 1
        return node

    def estimate_value(self, node):  # New function to replace simulation this is wrong
        """
        current_state = node.state
        state_buffer = [current_state for _ in range(6)]
        preprocessed_state = supplementary.preprocess_board_state_sequence(state_buffer)
        return value_net(preprocessed_state)[-1].item()
        """
        return self.simulate(node, max_depth=10)

    def simulate(self, node, max_depth=10):
        current_state = deepcopy(node.state)
        depth = 0
        
        while supplementary.check_goal(current_state['ball_pos']) is None and depth < max_depth:
            # Random action for each player
            available_actions = ['MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN']
            action_a1 = random.choice(available_actions)
            action_a2 = random.choice(available_actions)
            action_b1 = random.choice(available_actions)
            action_b2 = random.choice(available_actions)
            
            # Adjust state based on actions
            current_state = supplementary.check_validity(action_a1, deepcopy(current_state), "A1")
            current_state = supplementary.check_validity(action_a2, current_state, "A2")
            current_state = supplementary.check_validity(action_b1, current_state, "B1")
            current_state = supplementary.check_validity(action_b2, current_state, "B2")
            
            # Handle the ball if it's in possession
            if current_state['ball_possession'] is not None:
                ball_actions = ['SHOOT_LEFT', 'SHOOT_RIGHT', 'SHOOT_UP', 'SHOOT_DOWN']
                ball_move = random.choice(ball_actions)
                current_state = supplementary.check_ball_validity(ball_move, current_state)
            
            depth += 1
        
        # Use the value net to estimate the value of the current state
        state_tensor = supplementary.preprocess_states(current_state)
        estimated_value = value_net(state_tensor)[-1].item()

        
        return estimated_value
    
    def backpropagation(self, node, score):
        while node:
            node.visits += 1
            node.value += score
            node = node.parent
    
    def choose_node(self, node, exploration_constant=5.0, epsilon=0.3):
        state_buffer = supplementary.build_state_buffer(node)
        preprocessed_state = supplementary.preprocess_board_state_sequence(state_buffer)
        last_sequence_scores = model(preprocessed_state).squeeze().detach().numpy()[-1]
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(node.children)
        
        # Decay epsilon after making a decision, ensuring it doesn't go below the minimum threshold
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        
        best_ucb = float('-inf')
        best_node = None
        for child in node.children:
            bias_index = supplementary.action_to_index(child.action)
            bias = last_sequence_scores[bias_index]

            ucb = float('inf')  # default value
            if child.visits > 0:
                exploration_bonus = exploration_constant * math.sqrt((math.log(node.visits)) / child.visits)
                ucb = child.value / child.visits + exploration_bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child
        return best_node


    



supplementary = Supplementary()
value_net = ValueNet()

model = PlayersTransformerSeq()
model.load_state_dict(torch.load('model_players.pth'))
model.eval()

ball_model = BallTransformerSeq()
ball_model.load_state_dict(torch.load('model_ball.pth'))
ball_model.eval()

env = SoccerEnv()
state = env.state
initial_matrix = supplementary.state_to_matrix(state)
print("Initial state:")
print(initial_matrix)

mcts = MCTS(initial_epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05)
new_state = mcts.search(state)

print("End state")
end_matrix = supplementary.state_to_matrix(new_state.state)
print(end_matrix)

print("testing next")
new_state = mcts.search(new_state.state)
end_matrix = supplementary.state_to_matrix(new_state.state)
print(end_matrix)

"""
env.play_game(mcts.search, buffer, num_games=1)
"""


learning_rate = 0.001

player_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
ball_optimizer = optim.Adam(ball_model.parameters(), lr=learning_rate)
value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

def identify_indices_of_next_state_player(next_state):
    indices = []

    for pos in next_state['player_positions'].values():
        indices.append((pos[1], pos[0]))
    
    return indices

def identify_indices_of_next_state_ball(next_state):
    pos = next_state['ball_pos']
    return [(pos[1], pos[0])]


def adjust_prediction_with_reward(logits, next_state, reward, scale_factor=1.0):
    adjusted_logits = logits.clone().view(env.GRID_HEIGHT, env.GRID_WIDTH)
    
    indices = identify_indices_of_next_state_player(next_state)
    
    for idx in indices:
        if 0 <= idx[0] < env.GRID_HEIGHT and 0 <= idx[1] < env.GRID_WIDTH:
            adjusted_logits[idx[0], idx[1]] += reward * scale_factor
    
    return adjusted_logits.view(-1)  # Flatten the tensor back to its original shape

def adjust_prediction_with_reward_ball(logits, next_state, reward, scale_factor=1.0):
    adjusted_logits = logits.clone().view(env.GRID_HEIGHT, env.GRID_WIDTH)
    
    indices = identify_indices_of_next_state_ball(next_state)
    
    for idx in indices:
        if 0 <= idx[0] < env.GRID_HEIGHT and 0 <= idx[1] < env.GRID_WIDTH:
            adjusted_logits[idx[0], idx[1]] += reward * scale_factor
    
    return adjusted_logits.view(-1)  # Flatten the tensor back to its original shape

#print(buffer.buffer)

print("playing games")
player_losses = []
ball_losses = []
value_net_losses = []
total_losses = []

buffer = ReplayBuffer(1000000)
n_game = 0
batch_size = 64

# Lists for storing processed states, next states, and rewards
states = []
rewards = []
states_ = []

for episode in range(2):
    # Remember the buffer length before adding new game sequences
    old_buffer_length = len(buffer.buffer)
    
    buffer, n_game = env.play_game(mcts.search, deepcopy(buffer), 1, num_games=1)

    buffer_length = len(buffer.buffer)

    if buffer_length - old_buffer_length >= 7:  
        for i in range(old_buffer_length, buffer_length - 6):  # Only process the newly added sequences
            current_game_id = buffer.buffer[i]['game_id']
            
            # Check if all states in the current window belong to the same game
            if all([item['game_id'] == current_game_id for item in buffer.buffer[i:i+6]]):
                state_sequence = [item['state'] for item in buffer.buffer[i:i+6]]
                states.append(supplementary.preprocess_board_state_sequence(state_sequence))
                rewards.append(buffer.buffer[i+5]['reward'])
                next_state = [buffer.buffer[i+6]['state']]
                states_.append(next_state)

        if len(states) >= batch_size:
            n_epochs = 1
            for i in range(n_epochs):
                total_loss = []

                batch_indices = np.random.choice(len(states), batch_size, replace=False)
                batch_states = [states[i] for i in batch_indices]
                batch_states_ = [states_[i] for i in batch_indices]
                batch_rewards = [rewards[i] for i in batch_indices]

                for state_seq, state_, reward in zip(batch_states, batch_states_, batch_rewards):
                    player_optimizer.zero_grad()
                    predicted_next_state = model(state_seq) # player
                    adjusted_next_state = adjust_prediction_with_reward(predicted_next_state[-1], state_[0], reward) # Added state_
                    loss = F.mse_loss(predicted_next_state[-1], adjusted_next_state)
                    loss.backward()
                    player_optimizer.step()
                    total_loss.append(loss)

                    ball_optimizer.zero_grad()
                    predicted_next_state = ball_model(state_seq) # player
                    adjusted_next_state = adjust_prediction_with_reward_ball(predicted_next_state[-1], state_[0], reward) # Added state_
                    loss = F.mse_loss(predicted_next_state[-1], adjusted_next_state)
                    loss.backward()
                    ball_optimizer.step()
                    total_loss.append(loss)

                    value_net.zero_grad()
                    predicted_state_value_tensor = value_net(state_seq)[-1].unsqueeze(0)  # Keeping it as a tensor of shape (1,)
                    reward_tensor = torch.tensor([reward], dtype=torch.float32)
                    loss = F.mse_loss(predicted_state_value_tensor, reward_tensor)
                    loss.backward()
                    value_optimizer.step()
                    total_loss.append(loss)
                
                print(f"game number {episode} player loss {total_loss[0]}, ball loss {total_loss[1]}, value net loss {total_loss[2]} total {total_loss[0]+total_loss[1]+total_loss[2]}")
                
                avg_player_loss = sum(loss for i, loss in enumerate(total_loss) if i % 3 == 0) / len(total_loss) * 3
                avg_ball_loss = sum(loss for i, loss in enumerate(total_loss) if i % 3 == 1) / len(total_loss) * 3
                avg_value_net_loss = sum(loss for i, loss in enumerate(total_loss) if i % 3 == 2) / len(total_loss) * 3
                avg_total_loss = (sum(total_loss) / len(total_loss)) / episode

                player_losses.append(avg_player_loss.detach().numpy())
                ball_losses.append(avg_ball_loss.detach().numpy())
                value_net_losses.append(avg_value_net_loss.detach().numpy())
                total_losses.append(avg_total_loss.detach().numpy())


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
#plt.plot(player_losses, label="Player Loss")
#plt.plot(ball_losses, label="Ball Loss")
#plt.plot(value_net_losses, label="Value Net Loss")
plt.plot(total_losses, label="Total Loss", linestyle="--", color="grey")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses Over Time")
plt.grid(True)
plt.show()


