from dm_control import suite
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control import viewer
# from soccer_body_components import create_soccer_environment
import numpy as np
from asset_components_new import create_flags_and_creatures
import math

blueprint = {
    '0': {
        # TODO: I think make this (0 0 0), handle shifts elsewhere outside.
        'position': (0.00, 0.00, -5.34), #yz swapped, 
        'rotation': (0.0, 0.0, 0.0), #yz swapped
        'size': (0.78, 0.44, 1.32), #yz swapped
        'parent_unique_id': None,
        'joint_type': None, 
        'joint_anchorpos': None, #yz swapped
        'joint_axis': None, #yz swapped
        'color': (27.00, 255.00, 233.00)
    },
    '1': {
        # TODO: think about direction of the offset (probably will depend on something). but for now for this test case,
        'position': tuple(np.subtract((0.00, 0.00, -5.34), (-0.37, 0.07, -4.02)))[:2] + (-4.02 + 0.42 + 0.00000001,), #yz swapped, then subtracted from the position of the parent segment, then z value replaced accordingly. TODO: verify that this works.. intuitively it's a bit sus
        'rotation': (0.0, 0.0, 0.0), #yz swapped
        'size': (0.60, 0.39, 0.42), #yz swapped
        'parent_unique_id': 1,
        'joint_type': 'hinge',
        'joint_anchorpos': (-0.48, 0.16, 1.00), #yz swapped
        'joint_axis': (0, 0, 1), #yz swapped
        'color': (187.00, 84.00, 213.00)
    }
}


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
xml_soccer, _ =  create_flags_and_creatures(num_creatures=1, blueprint=blueprint)
print(xml_soccer)

# Load and render the environment
load_and_render_soccer_env(xml_soccer)
