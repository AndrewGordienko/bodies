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



class Supplementary():
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
                # Convert environment coordinates to grid indices
                grid_x, grid_y = self.convert_position_to_grid_index(pos)
                grid[grid_y, grid_x] = player_mapping[player]
        
        # Place the ball on the grid
        ball_pos = positions['ball']
        ball_x, ball_y = self.convert_position_to_grid_index(ball_pos)
        grid[ball_y, ball_x] += ball_value

        return grid

    def convert_position_to_grid_index(self, pos):
        # Assuming pos is a numpy array or a list/tuple of coordinates (x, y)
        # Convert environment coordinates to grid indices
        # This conversion logic might need adjustments based on your environment's coordinate system
        grid_x = int((pos[0] + GRID_WIDTH / 2) // 1)  # Example conversion logic
        grid_y = int((pos[1] + GRID_HEIGHT / 2) // 1) # Example conversion logic
        return grid_x, grid_y

    def process_grid_for_networks(self, grid):
        # Flatten the grid
        flat_grid = grid.flatten()
        
        # Convert to tensor of type Long
        input_tensor = torch.tensor(flat_grid, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        
        return input_tensor
    
    def preprocess_board_state_sequence(self, states):
        state_sequence = []

        print("states")
        print(states)

        if len(states) < 6:
            while state_sequence < 6:
                state_sequence.append(states[0])
        
        if state_sequence != []:
            states = state_sequence
        state_sequence = []

        for state in states:
            players_flat_positions = []
            for i in range(4):  # Assuming 4 players as in the original
                ii = np.where(state == i + 1)
                if ii[0].size > 0:  # Check if the player is found in the state
                    players_flat_positions.append(ii[0][0] * GRID_HEIGHT + ii[1][0])
                else:
                    players_flat_positions.append(-1)  # Placeholder if player not found

            test = np.where(state >= 10)



            ball_x, ball_y = np.where(state >= 10)
            if ball_x.size > 0:  # Check if the ball is found
                ball_flat_position = ball_x[0] * GRID_HEIGHT + ball_y[0]
                ball_possession = 1
            
            ball_x, ball_y = np.where(state == 10)
            if ball_x.size > 0:  # Check if the ball is found
                ball_flat_position = ball_x[0] * GRID_HEIGHT + ball_y[0]
                ball_possession = 0

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
        # Convert state indices to coordinates

        print("state")
        print(state)
        print("player")
        print(player)
        initial_pos = np.array(np.where(state == player)).T[0]
        initial_ball_pos = np.array(np.where(state >= 10)).T[0]  # Assuming '10' represents the ball

        player_pos = np.array(np.where(state == player)).T[0]
        ball_pos = np.array(np.where(state >= 10)).T[0]  # Assuming '10' represents the ball

        print(player_pos, ball_pos)
        if abs(player_pos[0] - ball_pos[0]) <= 1 and abs(player_pos[1] - ball_pos[1]) <= 1: # snap them together
            state[player_pos[0], player_pos[1]] += 10
            state[ball_pos[0], ball_pos[1]] = 0
        
        # Update the ball's position based on the player's movement if the player possesses the ball
        if state.max() > 10:
            if action == 'MOVE_LEFT' and player_pos[0] > 0:
                ball_pos[0] -= 1
            elif action == 'MOVE_RIGHT' and player_pos[0] < GRID_WIDTH - 1:
                ball_pos[0] += 1
            elif action == 'MOVE_UP' and player_pos[1] > 0:
                ball_pos[1] -= 1
            elif action == 'MOVE_DOWN' and player_pos[1] < GRID_HEIGHT - 1:
                ball_pos[1] += 1
        
        # Update the player's position based on the action
        if action == 'MOVE_LEFT' and player_pos[0] > 0:
            player_pos[0] -= 1
        elif action == 'MOVE_RIGHT' and player_pos[0] < GRID_WIDTH - 1:
            player_pos[0] += 1
        elif action == 'MOVE_UP' and player_pos[1] > 0:
            player_pos[1] -= 1
        elif action == 'MOVE_DOWN' and player_pos[1] < GRID_HEIGHT - 1:
            player_pos[1] += 1
         
        # Update state with new player and ball positions
        state[initial_pos[0]][initial_pos[1]] = 0
        state[initial_ball_pos[0]][initial_ball_pos[1]] = 0
        
        state[player_pos[0]][player_pos[1]] = player
        state[ball_pos[0]][ball_pos[1]] += 10 

        return state


        

    def check_goal(self, ball_pos):
        # Define the goal area dimensions and position
        net_height = 3
        net_width = 1  # Assuming the width of the goal post is always 1
        net_top_position = (GRID_HEIGHT - net_height) // 2

        print("check goal place")
        print(ball_pos)

        # Extract the y-coordinate from the ball position tuple
        ball_y = ball_pos[0]

        # Check if ball is within the vertical bounds of the goal
        if net_top_position <= ball_y <= net_top_position + net_height:
            ball_x = ball_pos[1]  # Extract the x-coordinate for horizontal position check
            if ball_x <= 1:  # Left side goal
                return 'B'
            elif ball_x >= GRID_WIDTH - net_width:  # Right side goal
                return 'A'
        return None


    def check_ball_validity(self, action, state):
        ball_pos = np.array(np.where(state >= 10)).T  # Assuming '10' represents the ball
        if ball_pos.size > 0:
            ball_pos = ball_pos[0]  # Take the first (and should be only) set of coordinates

            # Update ball position based on the action
            if action == 'SHOOT_LEFT' and ball_pos[1] > 1:
                state[ball_pos[0], ball_pos[1]] = 0
                ball_pos[1] -= 2
            elif action == 'SHOOT_RIGHT' and ball_pos[1] < GRID_WIDTH - 2:
                state[ball_pos[0], ball_pos[1]] = 0
                ball_pos[1] += 2
            elif action == 'SHOOT_UP' and ball_pos[0] > 1:
                state[ball_pos[0], ball_pos[1]] = 0
                ball_pos[0] -= 2
            elif action == 'SHOOT_DOWN' and ball_pos[0] < GRID_HEIGHT - 2:
                state[ball_pos[0], ball_pos[1]] = 0
                ball_pos[0] += 2

            # Update state with new ball position
            if 0 <= ball_pos[0] < GRID_HEIGHT and 0 <= ball_pos[1] < GRID_WIDTH:  # Ensure within bounds
                state[ball_pos[0], ball_pos[1]] += 10  # Update with ball value

        return state
    
    def build_state_buffer(self, node):
        state_buffer = [node.state]
        temp_node = node
        while len(state_buffer) < 6 and temp_node.parent:
            state_buffer.insert(0, temp_node.parent.state)
            temp_node = temp_node.parent
        while len(state_buffer) < 6:
            state_buffer.insert(0, state_buffer[0])
        return state_buffer

                
    
                
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

        # ii = np.where(state == i + 1)
        pos_A1 = np.where(self.state == 1)
        pos_A2 = np.where(self.state == 2)
        pos_B1 = np.where(self.state == 3)
        pos_B2 = np.where(self.state == 4)

        # pos_A1 = self.state["player_positions"]["A1"]
        # pos_A2 = self.state["player_positions"]["A2"]
        # pos_B1 = self.state["player_positions"]["B1"]
        # pos_B2 = self.state["player_positions"]["B2"]

        possession, team = None, None

        ball_position = np.where(self.state >= 10)
        ball_value_coordinate = self.state[ball_position[0][0]][ball_position[1][0]]

        if ball_value_coordinate == 11: possession, team = pos_A1, True
        if ball_value_coordinate == 12: possession, team = pos_A2, True
        if ball_value_coordinate == 13: possession, team = pos_B1, False
        if ball_value_coordinate == 14: possession, team = pos_B2, False
        
        # if self.state["ball_possession"] == "A1": possession, team = pos_A1, True
        # if self.state["ball_possession"] == "A2": possession, team = pos_A2, True
        # if self.state["ball_possession"] == "B1": possession, team = pos_B1, False
        # if self.state["ball_possession"] == "B2": possession, team = pos_B2, False

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
                                state = supplementary.check_validity(action_a1, deepcopy(self.state), 1)
                                state = supplementary.check_validity(action_a2, state, 2)
                                state = supplementary.check_validity(action_b1, state, 3)
                                state = supplementary.check_validity(action_b2, state, 4)
                                state = supplementary.check_ball_validity(ball_move, state)

                                temporary_node = Node(self, deepcopy(state))
                                temporary_node.action = (action_a1, action_a2, action_b1, action_b2, ball_move)
                                list_of_children.append(temporary_node)
                        else:
                            state = supplementary.check_validity(action_a1, deepcopy(self.state), 1)
                            state = supplementary.check_validity(action_a2, state, 2)
                            state = supplementary.check_validity(action_b1, state, 3)
                            state = supplementary.check_validity(action_b2, state, 4)

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
        #print("here")
        starting_node = Node(None, state)
        starting_node.create_children()

        for i in range(self.search_length):
            new_node = self.selection(starting_node)

            #print("selection state printed below")
            #print(new_node.state)
            #print("selection state printed above")

            score = self.estimate_value(new_node)
            self.backpropagation(new_node, score)
        
        best_action_value = float("-inf")
        best_child = None
        for child in starting_node.children:
            value = child.value / (child.visits + 1)
            if value > best_action_value:
                best_child = child
                best_action_value = value
        
        #print("full search")
        #print(state)
        #print(starting_node)
        #print(best_child)
        #print(best_child.state)
        return best_child.state
    
    def selection(self, node):
        depth = 0
        ball_pos = np.where(node.state >= 10)
        while supplementary.check_goal(ball_pos) == None and depth < self.depth:
            if not node.children or node.visits == 0:
                print("reached here lets see whats happening")

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
        
        ball_pos = np.where(current_state >= 10)
        while supplementary.check_goal(ball_pos) is None and depth < max_depth:
            # Random action for each player
            available_actions = ['MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN']
            action_a1 = random.choice(available_actions)
            action_a2 = random.choice(available_actions)
            action_b1 = random.choice(available_actions)
            action_b2 = random.choice(available_actions)
            
            # Adjust state based on actions

            current_state = supplementary.check_validity(action_a1, deepcopy(current_state), 1)
            current_state = supplementary.check_validity(action_a2, current_state, 2)
            current_state = supplementary.check_validity(action_b1, current_state, 3)
            current_state = supplementary.check_validity(action_b2, current_state, 4)
            
            # Handle the ball if it's in possession
            
            if np.where(current_state >= 10) is not None:
                ball_actions = ['SHOOT_LEFT', 'SHOOT_RIGHT', 'SHOOT_UP', 'SHOOT_DOWN']
                ball_move = random.choice(ball_actions)
                current_state = supplementary.check_ball_validity(ball_move, current_state)
            
            depth += 1
        
        # Use the value net to estimate the value of the current state
        state_buffer = [current_state for _ in range(6)] # this needs to be changed so it actually goes back
        state_tensor = supplementary.preprocess_board_state_sequence(state_buffer)
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


                

supplementary = Supplementary()
value_net = ValueNet()
mcts = MCTS()

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
        self.positions = {}

    def initialize_episode(self, physics):
        self.positions = self.extract_positions(physics)
        self.update_grid_representation(physics)

    def get_observation(self, physics):
        # Custom observation based on your task
        return {}

    def get_reward(self, physics):
        # Custom reward based on your task
        return 0

    def after_step(self, physics):
        targets = self.generate_target_positions()
        self.apply_action(physics, targets)
        self.update_grid_representation(physics)

    def update_grid_representation(self, physics):
        # Update any representations you maintain for the environment
        pass

    def extract_positions(self, physics):
        positions = {}
        entity_names = ['team_red_player_1', 'team_red_player_2', 'team_blue_player_1', 'team_blue_player_2', 'ball']
        for name in entity_names:
            positions[name] = physics.named.data.xpos[name][:2]  # Extracting x, y positions
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

def load_and_render_soccer_env(xml_string):
    # Parse the XML string to a MuJoCo model
    model = mujoco.wrapper.MjModel.from_xml_string(xml_string)
    physics = mujoco.Physics.from_model(model)
    
    # Create an instance of the environment
    task = CustomSoccerEnv(xml_string)
    env = control.Environment(physics, task, time_limit=20)
    
    # Define a dummy policy that does nothing (for demonstration purposes)
    def policy(time_step):
        action_spec = env.action_spec()
        return np.zeros(action_spec.shape)
    
    # Assuming `physics` is your MuJoCo physics object
    positions = extract_positions(physics)

    grid_representation = supplementary.create_grid_representation(positions)

    # Assuming `physics` is your MuJoCo physics object
    positions = extract_positions(physics)
    
    # Create initial grid representation and update the buffer
    initial_grid = supplementary.create_grid_representation(positions)

    #state_sequence_tensor = supplementary.preprocess_board_state_sequence(initial_grid) # state has to be set to six for this line to work
    #print(state_sequence_tensor.shape)

    #print(player_model.forward(state_sequence_tensor))
    #print(ball_model.forward(state_sequence_tensor))
    #print(value_net.forward(state_sequence_tensor))

    print("hello up")
    result = mcts.search(initial_grid)
    print("hello down")
    print(result)
    

    # Use the dm_control viewer to render the environment
    viewer.launch(env, policy=policy)
    
    return physics  # Return the physics object for further use


# Generate the XML for the soccer environment
xml_soccer = create_soccer_environment()

# Load and render the environment, and receive the physics object
physics = load_and_render_soccer_env(xml_soccer)


