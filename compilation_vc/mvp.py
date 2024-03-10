# scuffed mvp with random policy

from dm_control import suite
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control import viewer
from soccer_body_components import create_soccer_environment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import math
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
        policy_logits = player_model.forward(preprocessed_state)[0].squeeze().detach().numpy() 
        policy_logits = policy_logits.reshape(GRID_WIDTH, GRID_HEIGHT)
        ball_policy_logits = ball_model.forward(preprocessed_state)[0].squeeze().detach().numpy() 
        ball_policy_logits = ball_policy_logits.reshape(GRID_WIDTH, GRID_HEIGHT)

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
        last_sequence_scores = player_model(preprocessed_state).squeeze().detach().numpy()[-1]
        
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

class Supplementary():
    def array_to_dict(self, array):
        player_positions = {}
        ball_pos = None
        ball_possession = None
        
        # Map numbers to player IDs
        player_id_map = {1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2'}
        
        for y, row in enumerate(array):
            for x, value in enumerate(row):
                if value in player_id_map:
                    # Assign player positions based on their number
                    player_positions[player_id_map[value]] = [x, y]
                    if value > 10:
                        # Indicates both the player and the ball are in this cell
                        ball_pos = [x, y]
                        ball_possession = player_id_map[value % 10]  # Use modulo to find out who has the ball
                elif value == 10:
                    # Assign ball position
                    ball_pos = [x, y]
        
        # Constructing the state dictionary
        state = {
            'player_positions': player_positions,
            'ball_pos': ball_pos,
            'ball_possession': ball_possession,
        }
        
        return state

    def convert_position_to_grid_index(self, pos):
        # Assuming pos is a numpy array or a list/tuple of coordinates (x, y)
        # Convert environment coordinates to grid indices
        # This conversion logic might need adjustments based on your environment's coordinate system
        grid_x = int((pos[0] + GRID_WIDTH / 2) // 1)  # Example conversion logic
        grid_y = int((pos[1] + GRID_HEIGHT / 2) // 1) # Example conversion logic
        return grid_x, grid_y
    
    def create_grid_representation(self, positions):
        # Initialize the grid with zeros
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        
        # Mapping for players to grid values
        player_mapping = {
            "team_red_player_1": 1,
            "team_red_player_2": 2,
            "team_blue_player_1": 3,
            "team_blue_player_2": 4,
        }
        ball_value = 10

        # Place players on the grid
        for player, pos in positions.items():
            if player in player_mapping:
                print(player, pos)
                # Convert environment coordinates to grid indices
                #grid_x, grid_y = self.convert_position_to_grid_index(pos)
                grid_x, grid_y = pos
                print(grid_x, grid_y)
                grid[int(grid_y)][int(grid_x)] = player_mapping[player]
        
        # Place the ball on the grid
        ball_pos = positions['ball']
        #ball_x, ball_y = self.convert_position_to_grid_index(ball_pos)
        ball_x, ball_y = ball_pos
        grid[int(ball_y), int(ball_x)] += ball_value

        return grid
    
    def state_to_matrix(self, state):
        matrix = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        player_mapping = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4}
        for player, position in state['player_positions'].items():
            matrix[position[1], position[0]] = player_mapping[player]
        matrix[state['ball_pos'][1]][state['ball_pos'][0]] += 10
        return matrix

    def preprocess_board_state_sequence(self, states):
        state_sequence = []
        for state in states:
            players_flat_positions = [pos[0] * GRID_HEIGHT + pos[1] for pos in state['player_positions'].values()]
            ball_x = min(int(state['ball_pos'][0]), GRID_WIDTH - 1)
            ball_y = min(int(state['ball_pos'][1]), GRID_HEIGHT - 1)
            ball_flat_position = ball_x * GRID_HEIGHT + ball_y
            ball_possession = 0 if state['ball_possession'] is None else 1
            state_sequence.append(players_flat_positions + [ball_flat_position, ball_possession])
        return torch.tensor([state_sequence], dtype=torch.long).squeeze(0)

    def get_surrounding_actions(self, player_pos, logits, is_team1=True):
        x, y = player_pos
        surrounding_values = {}

        if x > 0: surrounding_values['MOVE_LEFT'] = logits[x-1, y]
        if x < GRID_WIDTH - 1: surrounding_values['MOVE_RIGHT'] = logits[x+1, y]
        if y > 0: surrounding_values['MOVE_UP'] = logits[x, y-1]
        if y < GRID_HEIGHT - 1: surrounding_values['MOVE_DOWN'] = logits[x, y+1]

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
        if x < GRID_WIDTH - 2: surrounding_values['SHOOT_RIGHT'] = logits[x+1, y]
        if y > 1: surrounding_values['SHOOT_UP'] = logits[x, y-1]
        if y < GRID_HEIGHT - 2: surrounding_values['SHOOT_DOWN'] = logits[x, y+1]

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
            elif action == 'MOVE_RIGHT' and initial_pos[0] < GRID_WIDTH - 1:
                state['ball_pos'][0] += 1
            elif action == 'MOVE_UP' and initial_pos[1] > 0:
                state['ball_pos'][1] -= 1
            elif action == 'MOVE_DOWN' and initial_pos[1] < GRID_HEIGHT - 1:
                state['ball_pos'][1] += 1
        
        # Update the player's position based on the action
        if action == 'MOVE_LEFT' and state['player_positions'][player][0] > 0:
            state['player_positions'][player][0] -= 1
        elif action == 'MOVE_RIGHT' and state['player_positions'][player][0] < GRID_WIDTH - 1:
            state['player_positions'][player][0] += 1
        elif action == 'MOVE_UP' and state['player_positions'][player][1] > 0:
            state['player_positions'][player][1] -= 1
        elif action == 'MOVE_DOWN' and state['player_positions'][player][1] < GRID_HEIGHT - 1:
            state['player_positions'][player][1] += 1

        return state
    
    def check_ball_validity(self, action, state):
        if action in ['SHOOT_LEFT', 'SHOOT_RIGHT', 'SHOOT_UP', 'SHOOT_DOWN']:
            if action == 'SHOOT_LEFT' and state['ball_pos'][0] > 1:
                state['ball_pos'][0] -= 2
            elif action == 'SHOOT_RIGHT' and state['ball_pos'][0] < GRID_WIDTH - 2:
                state['ball_pos'][0] += 2
            elif action == 'SHOOT_UP' and state['ball_pos'][1] > 1:
                state['ball_pos'][1] -= 2
            elif action == 'SHOOT_DOWN' and state['ball_pos'][1] < GRID_HEIGHT - 2:
                state['ball_pos'][1] += 2

            state['ball_possession'] = None
            for player in state['player_positions']:
                if state['player_positions'][player] == [state['ball_pos'][0], state['ball_pos'][1]]:
                    state['ball_possession'] = player

        return state

    def check_goal(self, ball_pos):
        net_height = 3
        net_width = 1  # Assuming the width of the goal post is always 1
        net_top_position = (GRID_HEIGHT - net_height) // 2

        if net_top_position <= ball_pos[1] <= net_top_position + net_height:
            if ball_pos[0] <= 1:  
                return 'B'
            elif ball_pos[0] >= GRID_WIDTH - 1:
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


supplementary = Supplementary()
mcts = MCTS()

value_net = ValueNet()
player_model = PlayersTransformerSeq()
player_model.load_state_dict(torch.load('model_players.pth'))
player_model.eval()

ball_model = BallTransformerSeq()
ball_model.load_state_dict(torch.load('model_ball.pth'))
ball_model.eval()

GRID_WIDTH = 9
GRID_HEIGHT = 7

FIELD_MIN_X = -4.5
FIELD_MAX_X = 4.5
FIELD_MIN_Y = -2
FIELD_MAX_Y = 2


class CustomSoccerEnv(base.Task):
    def __init__(self, xml_string):
        super().__init__()
        self.xml_string = xml_string
        # Define the action spec for your environment


    def initialize_episode(self, physics):
        self.positions = self.extract_positions(physics)
        self.update_grid_representation(physics)

    def get_observation(self, physics):
        # Custom observation based on your task
        return {}

    def get_reward(self, physics):
        # Custom reward based on your task
        return 0

    """
    def after_step(self, physics):
        targets = self.generate_target_positions()
        self.apply_action(physics, targets)
        self.update_grid_representation(physics)
    """

    def update_grid_representation(self, physics):
        # Update any representations you maintain for the environment
        pass

    def extract_positions(self, physics):
        positions = {}
        entity_names = ['team_red_player_1', 'team_red_player_2', 'team_blue_player_1', 'team_blue_player_2', 'ball']
        for name in entity_names:
            positions[name] = abs(physics.named.data.xpos[name][:2])  # Extracting x, y positions
        print("Extracted positions:", positions)
        return positions

    def generate_target_positions(self):
        targets = {}
        for player_name, current_pos in self.positions.items():
            if 'player' in player_name:  # Assuming you only want to move players, not the ball
                grid_x, grid_y = self.convert_position_to_grid_index(current_pos)
                target_grid_x, target_grid_y = self.get_random_adjacent_position((grid_x, grid_y))
                targets[player_name] = (target_grid_x, target_grid_y)
                print(f"Generated target for {player_name}: ({target_grid_x}, {target_grid_y})")
        return targets

    def apply_action(self, physics, targets):
        for player_name, (target_grid_x, target_grid_y) in targets.items():
            target_world_pos = self.convert_grid_index_to_position((target_grid_x, target_grid_y))
            self.move_player_to(physics, player_name, target_world_pos)
    
    def after_step(self, physics):
        # This method is correctly implemented
        self.apply_random_actions(physics)

    def apply_random_actions(self, physics):
        # Correctly access action_spec from the environment
        action_spec = self.action_spec(physics)

        # Generate random actions within the allowed range
        random_actions = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

        # Apply the random actions to the environment
        physics.set_control(random_actions)

    def move_player_to(self, physics, player_name, new_position):
        body_id = physics.model.name2id(player_name, 'body')
        addr = physics.model.body_jntadr[body_id]
        if (addr + 2) <= len(physics.data.qpos):
            physics.data.qpos[addr:addr+2] = new_position[:2]
            self.positions[player_name] = new_position
            print(f"Moved {player_name} to {new_position}")
        else:
            print(f"Error: Unable to move {player_name}. Address out of bounds.")

    def get_random_adjacent_position(self, position):
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while True:
            dx, dy = random.choice(movements)
            new_x, new_y = position[0] + dx, position[1] + dy
            if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
                return new_x, new_y

    def convert_position_to_grid_index(self, pos):
        grid_x = int((pos[0] + GRID_WIDTH / 2) // 1)
        grid_y = int((pos[1] + GRID_HEIGHT / 2) // 1)
        return grid_x, grid_y

    def convert_grid_index_to_position(self, grid_index):
        x = grid_index[0] * 1 - GRID_WIDTH / 2 + 0.5
        y = grid_index[1] * 1 - GRID_HEIGHT / 2 + 0.5
        return np.array([x, y, 0])  # Keeping z-coordinate as 0 for horizontal movement


def extract_positions(physics):
    # Names of the objects in the MuJoCo model
    player_names = [
        "team_red_player_1", "team_red_player_2",
        "team_blue_player_1", "team_blue_player_2"
    ]
    ball_name = "ball"

    # Dictionary to hold the positions
    positions = {}

    # Extract positions for each player
    for name in player_names:
        # Get the x, y position of the player
        position = physics.named.data.xpos[name][:2]  # Assuming we're only interested in the x, y coordinates
        positions[name] = position

    # Extract position for the ball
    ball_position = physics.named.data.xpos[ball_name][:2]  # Assuming we're only interested in the x, y coordinates
    positions[ball_name] = ball_position

    return positions

def choose_action(observation):
    # Convert observation to a NumPy array if it's a list of NumPy arrays
    if isinstance(observation, list):
        observation = np.array(observation)
    
    # Ensure observation is a 2D array for batch processing (if it's not already)
    if observation.ndim == 1:
        observation = np.expand_dims(observation, axis=0)

    # Convert observation to a tensor
    state = torch.tensor(observation, dtype=torch.float).to(DEVICE)
    
    dist = actor(state)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(axis=-1)
    value = critic.forward(state)
    
    action = action.cpu().numpy()
    log_prob = log_prob.detach().cpu().numpy()
    value = torch.squeeze(value).item()

    return action, log_prob, value

positions = None
import time

def load_and_render_soccer_env(xml_string, positions=None):
    # Parse the XML string to a MuJoCo model
    model = mujoco.wrapper.MjModel.from_xml_string(xml_string)
    physics = mujoco.Physics.from_model(model)
    
    # Create an instance of the environment
    task = CustomSoccerEnv(xml_string)
    env = control.Environment(physics, task, time_limit=20)
    
    # Define a dummy policy that does nothing (for demonstration purposes)

    
    # def policy(time_step):

    #     action_spec = env.action_spec()


    #     observation = np.zeros(41)
    #     action, _, _ = choose_action(observation) # observation needs to be collected sensors from the parts, and also coordinates of the target beacon

    #     print(action)
    #     # print("action")
    #     # print(action)
    #     # print("action spec")
    #     # print(action_spec.shape)    
    #     # print("herehehehehe")
    #     return np.zeros(action_spec.shape)
    
    def policy(time_step):
        action_spec = env.action_spec()

        # Generate random actions within the valid range
        random_actions = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

        return random_actions
    

    # def policy(time_step):
    #     action_spec = env.action_spec()
    #     print("action spec")
    #     print(action_spec)
    #     print("--")
    #     print(np.zeros(action_spec.shape))
    #     return [np.zeros(action_spec.shape)]


    # Assuming `physics` is your MuJoCo physics object
    if positions == None:
        positions = extract_positions(physics)

    print("racial slur")
    print(positions)
    
    # Create initial grid representation and update the buffer
    initial_grid = supplementary.create_grid_representation(positions)

    initial_grid[0][0] = 0
    initial_grid[3][4] = 10  # Assuming '10' uniquely identifies the ball


    dictionary_state = supplementary.array_to_dict(initial_grid)

    print("initial grid")
    print(initial_grid)
    print(dictionary_state)

    #state_sequence_tensor = supplementary.preprocess_board_state_sequence(initial_grid) # state has to be set to six for this line to work
    #print(state_sequence_tensor.shape)

    #print(player_model.forward(state_sequence_tensor))
    #print(ball_model.forward(state_sequence_tensor))
    #print(value_net.forward(state_sequence_tensor))

    print("hello up")
    result = mcts.search(dictionary_state)
    print(result)
    print("hello down")
    #print(result)
    

    # Use the dm_control viewer to render the environment
    viewer.launch(env, policy=policy)
    
    return physics  # Return the physics object for further use

class actor_network(nn.Module):
    def __init__(self, input_shape, action_space):
        super().__init__()

        self.std = 0.5

        self.fc1 = nn.Linear(input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, action_space)

        self.log_std = nn.Parameter(torch.ones(1, action_space) * 0.01)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

        self.log_std_min = -20  # min bound for log standard deviation
        self.log_std_max = 2    # max bound for log standard deviation


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
    def __init__(self, input_shape):
        super().__init__()
        # Unpack the input_shape tuple when passing it to nn.Linear
        self.fc1 = nn.Linear(input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")
LEARNING_RATE = 0.0003

# agent = Agent(n_actions=ACTION_DIMS, input_dims=[41], env=env)  
"""
def __init__(self, n_actions, input_dims, env):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 10
        self.gae_lambda = 0.95

        # Use input_dims directly
        print("input dims")
        print(input_dims)
        single_creature_obs_shape = (input_dims)

        self.actor = actor_network(single_creature_obs_shape, n_actions)
        self.critic = critic_network(single_creature_obs_shape)  # Updated line
        self.memory = PPOMemory(BATCH_SIZE)
"""

actor = actor_network(41, 12).to(DEVICE)
critic = critic_network(41).to(DEVICE)
import os
def load_models(directory="./", prefix="PPO"):
        actor_path = os.path.join(directory, f"{prefix}_Actor.pth")
        critic_path = os.path.join(directory, f"{prefix}_Critic.pth")
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            actor.load_state_dict(torch.load(actor_path))
            critic.load_state_dict(torch.load(critic_path))
            print(f"Models loaded: {actor_path} and {critic_path}")
        else:
            print("Model loading failed. Files do not exist.")

load_models()
# Generate the XML for the soccer environment
xml_soccer = create_soccer_environment()

# Load and render the environment, and receive the physics object
physics = load_and_render_soccer_env(xml_soccer)


