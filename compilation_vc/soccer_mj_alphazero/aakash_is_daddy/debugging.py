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
        self.positions = 0

    def get_observation(self, physics):
        # Custom observation based on your task
        return {}

    def get_reward(self, physics):
        # Custom reward based on your task
        return 0


def load_and_render_soccer_env(xml_string, positions=None):
    # Parse the XML string to a MuJoCo model
    model = mujoco.wrapper.MjModel.from_xml_string(xml_string)
    physics = mujoco.Physics.from_model(model)
    
    # Create an instance of the environment
    task = CustomSoccerEnv(xml_string)
    env = control.Environment(physics, task, time_limit=20)

    global x
    x = 10
    
    def policy(time_step):
        print(y)
        action_spec = env.action_spec()
        print("action spec")
        print(action_spec)
        print("--")
        print(np.zeros(action_spec.shape))
        return np.zeros(action_spec.shape)
    

    # Use the dm_control viewer to render the environment
    viewer.launch(env, policy)
    
    return physics  # Return the physics object for further use

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")
LEARNING_RATE = 0.0003

# Generate the XML for the soccer environment
xml_soccer = create_soccer_environment()

# Load and render the environment, and receive the physics object
physics = load_and_render_soccer_env(xml_soccer)


