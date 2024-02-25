from dm_control import suite
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control import viewer
# from soccer_body_components import create_soccer_environment
import numpy as np
from asset_components import create_ant_model
import math

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

def load_and_render_soccer_env(xml_string):
    # Parse the XML string to a MuJoCo model
    model = mujoco.wrapper.MjModel.from_xml_string(xml_string)
    physics = mujoco.Physics.from_model(model)
    
    # Create an instance of the environment
    task = CustomSoccerEnv(xml_string)
    env = control.Environment(physics, task, time_limit=20)
    
    # Initialize a step counter
    step_counter = 0

    # Define a dummy policy that does nothing (for demonstration purposes)
    def policy(time_step):
        nonlocal step_counter
        action_spec = env.action_spec()
        # return np.zeros(action_spec.shape)
        # return 0.1 * np.ones(action_spec.shape) 
        action = -200 * math.sin(0.005 * step_counter) * np.ones(action_spec.shape) 
        # Increment the step counter
        step_counter += 1
        return action
    

    
    # Use the dm_control viewer to render the environment
    viewer.launch(env, policy=policy)

# Generate the XML for the soccer environment
# xml_soccer = create_soccer_environment()
# xml_soccer =  create_ant_model(num_creatures=9)
xml_soccer, _ =  create_ant_model(num_creatures=1)

print(xml_soccer)
# Load and render the environment
load_and_render_soccer_env(xml_soccer)
