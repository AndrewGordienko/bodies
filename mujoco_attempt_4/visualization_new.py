from dm_control import suite
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control import viewer
# from soccer_body_components import create_soccer_environment
import numpy as np
from asset_components_new import create_flags_and_creatures
import math
import json

# Given JSON input
json_input = [
  {
    "UniqueId": 0,
    "TypeId": 1,
    "Position": {"x": 0.0, "y": -5.33999634, "z": 0.0},
    "Rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
    "Size": {"x": 0.7817855, "y": 1.3171674, "z": 0.436341643},
    "ParentUniqueId": None,
    "JointType": None,
    "JointAnchorPos": None,
    "JointAxis": None,
    "Color": {"x": 27.0, "y": 255.0, "z": 233.0}
  },
  {
    "UniqueId": 1,
    "TypeId": 3,
    "Position": {"x": -0.373039246, "y": -4.022829, "z": 0.06939888},
    "Rotation": {"x": 0.0008985509, "y": 0.00109280727, "z": 0.004466458},
    "Size": {"x": 0.595344, "y": 0.419512331, "z": 0.388579845},
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {"x": -0.477163136, "y": 0.99999994, "z": 0.159047112},
    "JointAxis": {"x": 0.0, "y": 1.0, "z": 0.0},
    "Color": {"x": 187.0, "y": 84.0, "z": 213.0}
  }
]

def convert_json_to_blueprint(json_input):
    blueprint = {}
    for item in json_input:
        unique_id = str(item["UniqueId"])
        position = (item["Position"]["x"], item["Position"]["z"], item["Position"]["y"])
        rotation = (item["Rotation"]["x"], item["Rotation"]["z"], item["Rotation"]["y"])
        size = (item["Size"]["x"], item["Size"]["z"], item["Size"]["y"])
        color = (item["Color"]["x"], item["Color"]["y"], item["Color"]["z"])

        # Handle parent and subtraction for position if needed
        if item["ParentUniqueId"] is not None and str(item["ParentUniqueId"]) in blueprint:
            parent_position = blueprint[str(item["ParentUniqueId"])]["position"]
            parent_size = blueprint[str(item["ParentUniqueId"])]["size"]
            # Subtract parent position from current position and adjust Z value
            adjusted_position = tuple(np.subtract(position[:2], parent_position[:2])) + (position[2] + parent_size[2] + 0.00000001,)
        else:
            adjusted_position = position

        if int(unique_id) != 1:
            blueprint[unique_id] = {
                'position': adjusted_position,
                'rotation': rotation,
                'size': size,
                'parent_unique_id': item["ParentUniqueId"],
                'joint_type': item.get("JointType"),
                'joint_anchorpos': None if not item.get("JointAnchorPos") else (
                    item["JointAnchorPos"]["x"], item["JointAnchorPos"]["z"], item["JointAnchorPos"]["y"]
                ),
                'joint_axis': None if not item.get("JointAxis") else (
                    item["JointAxis"]["x"], item["JointAxis"]["z"], item["JointAxis"]["y"]
                ),
                'color': color
            }
        else:
            blueprint[1] = {
                'position': tuple(np.subtract((0.00, 0.00, -5.34), (-0.37, 0.07, -4.02)))[:2] + (-4.02 + 0.42 + 0.00000001,),
                'rotation': rotation,
                'size': size,
                'parent_unique_id': item["ParentUniqueId"],
                'joint_type': item.get("JointType"),
                'joint_anchorpos': None if not item.get("JointAnchorPos") else (
                    item["JointAnchorPos"]["x"], item["JointAnchorPos"]["z"], item["JointAnchorPos"]["y"]
                ),
                'joint_axis': None if not item.get("JointAxis") else (
                    item["JointAxis"]["x"], item["JointAxis"]["z"], item["JointAxis"]["y"]
                ),
                'color': color
            }

    return blueprint

blueprint = convert_json_to_blueprint(json_input)

manual_blueprint = {
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
