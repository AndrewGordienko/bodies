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

class StateSequenceBuffer:
    def __init__(self, sequence_length=6, grid_height=7, grid_width=9):
        self.buffer = np.zeros((sequence_length, grid_height * grid_width), dtype=int)
        self.sequence_length = sequence_length
        self.grid_height = grid_height
        self.grid_width = grid_width

    def update_buffer(self, new_grid):
        # Flatten the new grid and append it to the buffer, then maintain the latest sequence_length states
        new_grid_flattened = new_grid.flatten()
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1, :] = new_grid_flattened

    def get_buffer_as_tensor(self):
        # Directly reshape the buffer to the correct shape expected by the transformer
        # The shape should be [sequence_length, batch_size=1, features]
        tensor = torch.tensor(self.buffer, dtype=torch.long)
        #tensor = tensor.unsqueeze(0)  # Add the batch dimension
        return tensor


# Adjust the function to use the updated buffer handling
def update_and_process_new_state(new_grid, buffer, player_model, ball_model, value_net):
    # Update the buffer with the new grid state
    buffer.update_buffer(new_grid)
    
    # Prepare the input tensor
    input_tensor = buffer.get_buffer_as_tensor()
    
    print(input_tensor.shape)
    # Feed the input tensor into the models
    player_output = player_model(input_tensor)
    ball_output = ball_model(input_tensor)
    value_output = value_net(input_tensor)

    # Print model outputs
    print("Player Network Output:", player_output)
    print("Ball Network Output:", ball_output)
    print("Value Network Output:", value_output.squeeze().item())



# Initialize the buffer
state_sequence_buffer = StateSequenceBuffer()


value_net = ValueNet()

player_model = PlayersTransformerSeq()
player_model.load_state_dict(torch.load('model_players.pth'))
player_model.eval()

ball_model = BallTransformerSeq()
ball_model.load_state_dict(torch.load('model_ball.pth'))
ball_model.eval()

GRID_WIDTH = 9
GRID_HEIGHT = 7

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

    grid_representation = create_grid_representation(positions)
    print(grid_representation)

    # Assuming `physics` is your MuJoCo physics object
    positions = extract_positions(physics)
    
    # Create initial grid representation and update the buffer
    initial_grid = create_grid_representation(positions)
    for _ in range(6):  # Initialize the buffer with 6 copies of the initial state
        update_and_process_new_state(initial_grid, state_sequence_buffer, player_model, ball_model, value_net)




    # Use the dm_control viewer to render the environment
    viewer.launch(env, policy=policy)
    
    return physics  # Return the physics object for further use

def create_grid_representation(positions):
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
            grid_x, grid_y = convert_position_to_grid_index(pos)
            grid[grid_y, grid_x] = player_mapping[player]
    
    # Place the ball on the grid
    ball_pos = positions['ball']
    ball_x, ball_y = convert_position_to_grid_index(ball_pos)
    grid[ball_y, ball_x] += ball_value

    return grid

def convert_position_to_grid_index(pos):
    # Assuming pos is a numpy array or a list/tuple of coordinates (x, y)
    # Convert environment coordinates to grid indices
    # This conversion logic might need adjustments based on your environment's coordinate system
    grid_x = int((pos[0] + GRID_WIDTH / 2) // 1)  # Example conversion logic
    grid_y = int((pos[1] + GRID_HEIGHT / 2) // 1) # Example conversion logic
    return grid_x, grid_y

def process_grid_for_networks(grid):
    # Flatten the grid
    flat_grid = grid.flatten()
    
    # Convert to tensor of type Long
    input_tensor = torch.tensor(flat_grid, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    return input_tensor


# Generate the XML for the soccer environment
xml_soccer = create_soccer_environment()

# Load and render the environment, and receive the physics object
physics = load_and_render_soccer_env(xml_soccer)

def set_positions(env, positions):
    """
    Set positions of players and ball in the MuJoCo environment.
    Args:
    - env: The MuJoCo environment instance.
    - positions: A numpy array or list of positions [(x1, y1), (x2, y2), ..., (ball_x, ball_y)]
                 The first N-1 are player positions, and the last one is the ball position.
    """
    physics = env.physics
    # Assuming the naming convention in your XML is 'player_1', 'player_2', ..., 'ball'
    for i, pos in enumerate(positions[:-1], start=1):
        player_name = f"player_{i}"
        physics.named.data.qpos[player_name][:2] = pos
    
    # Set ball position
    ball_name = "ball"
    physics.named.data.qpos[ball_name][:2] = positions[-1]

    # Manually step the environment to apply the changes
    physics.after_reset()
