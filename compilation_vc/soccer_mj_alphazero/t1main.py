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
    
    def preprocess_board_state_sequence(self, state):
        states = [state] * 6

        state_sequence = []
        for state in states:
            players_flat_positions = []
            for i in range(4):  # Assuming 4 players as in the original
                ii = np.where(state == i + 1)
                if ii[0].size > 0:  # Check if the player is found in the state
                    players_flat_positions.append(ii[0][0] * GRID_HEIGHT + ii[1][0])
                else:
                    players_flat_positions.append(-1)  # Placeholder if player not found

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
    
                
    
                

supplementary = Supplementary()
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
        self.xml_string = xml_string
        super().__init__()

    def initialize_episode(self, physics):
        # This method is called at the start of each episode, you can reset the environment here
        pass

    def get_observation(self, physics):
        # Here, return an observation based on the current state of the physics
        return {}

    def get_reward(self, physics):
        # Define and return a reward based on the current state of the environment
        return 0


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
    print(positions)

    grid_representation = supplementary.create_grid_representation(positions)
    print(grid_representation)

    # Assuming `physics` is your MuJoCo physics object
    positions = extract_positions(physics)
    
    # Create initial grid representation and update the buffer
    initial_grid = supplementary.create_grid_representation(positions)

    print("--")
    print(initial_grid)

    state_sequence_tensor = supplementary.preprocess_board_state_sequence(initial_grid)
    print(state_sequence_tensor.shape)

    #print(player_model.forward(state_sequence_tensor))
    #print(ball_model.forward(state_sequence_tensor))
    #print(value_net.forward(state_sequence_tensor))
    

    # Use the dm_control viewer to render the environment
    viewer.launch(env, policy=policy)
    
    return physics  # Return the physics object for further use


# Generate the XML for the soccer environment
xml_soccer = create_soccer_environment()

# Load and render the environment, and receive the physics object
physics = load_and_render_soccer_env(xml_soccer)

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
    print(positions)

    grid_representation = supplementary.create_grid_representation(positions)
    print(grid_representation)

    # Assuming `physics` is your MuJoCo physics object
    positions = extract_positions(physics)
    
    # Create initial grid representation and update the buffer
    initial_grid = supplementary.create_grid_representation(positions)

    print("--")
    print(initial_grid)

    state_sequence_tensor = supplementary.preprocess_board_state_sequence(initial_grid)
    print(state_sequence_tensor.shape)

    #print(player_model.forward(state_sequence_tensor))
    #print(ball_model.forward(state_sequence_tensor))
    #print(value_net.forward(state_sequence_tensor))
    

    # Use the dm_control viewer to render the environment
    viewer.launch(env, policy=policy)
    
    return physics  # Return the physics object for further use


# Generate the XML for the soccer environment
xml_soccer = create_soccer_environment()

# Load and render the environment, and receive the physics object
physics = load_and_render_soccer_env(xml_soccer)


