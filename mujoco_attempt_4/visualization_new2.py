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


json_input_3_seg_worm = [
  {
    "UniqueId": 0,
    "TypeId": 1,
    "Position": {
      "x": 0.0,
      "y": -5.33999634,
      "z": 0.0,
      "normalized": {
        "x": 0.0,
        "y": -1.0,
        "z": 0.0,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 5.33999634,
      "sqrMagnitude": 28.51556
    },
    "LocalPosition": {
      "x": 0.0,
      "y": -5.33999634,
      "z": 0.0,
      "normalized": {
        "x": 0.0,
        "y": -1.0,
        "z": 0.0,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 5.33999634,
      "sqrMagnitude": 28.51556
    },
    "Rotation": {
      "x": 0.000158063776,
      "y": 0.000541241141,
      "z": 0.00360564,
      "normalized": {
        "x": 0.04331154,
        "y": 0.148307145,
        "z": 0.987992465,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.003649461,
      "sqrMagnitude": 1.33185658E-05
    },
    "LocalRotation": {
      "x": 0.000158063776,
      "y": 0.000541241141,
      "z": 0.00360564,
      "normalized": {
        "x": 0.04331154,
        "y": 0.148307145,
        "z": 0.987992465,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.003649461,
      "sqrMagnitude": 1.33185658E-05
    },
    "Size": {
      "x": 0.7242545,
      "y": 1.47805929,
      "z": 0.681036532,
      "normalized": {
        "x": 0.406588584,
        "y": 0.8297664,
        "z": 0.3823265,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 1.78129566,
      "sqrMagnitude": 3.1730144
    },
    "ParentUniqueId": None,
    "JointType": None,
    "JointAnchorPos": None,
    "JointAxis": None,
    "Color": {
      "x": 26.0,
      "y": 254.0,
      "z": 233.0,
      "normalized": {
        "x": 0.07521837,
        "y": 0.7348256,
        "z": 0.6740723,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 345.660248,
      "sqrMagnitude": 119481.0
    }
  },
  {
    "UniqueId": 1,
    "TypeId": 2,
    "Position": {
      "x": 0.345495462,
      "y": -3.861915,
      "z": 0.108317375,
      "normalized": {
        "x": 0.08907159,
        "y": -0.995633662,
        "z": 0.0279251151,
        "normalized": {
          "x": 0.0890715942,
          "y": -0.9956337,
          "z": 0.0279251169,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 3.87885141,
      "sqrMagnitude": 15.0454884
    },
    "LocalPosition": {
      "x": 0.345495462,
      "y": -3.861915,
      "z": 0.108317375,
      "normalized": {
        "x": 0.08907159,
        "y": -0.995633662,
        "z": 0.0279251151,
        "normalized": {
          "x": 0.0890715942,
          "y": -0.9956337,
          "z": 0.0279251169,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 3.87885141,
      "sqrMagnitude": 15.0454884
    },
    "Rotation": {
      "x": 0.00431122072,
      "y": -0.00127039349,
      "z": -0.005556598,
      "normalized": {
        "x": 0.60324055,
        "y": -0.17775774,
        "z": -0.777497947,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.00714676874,
      "sqrMagnitude": 5.10763057E-05
    },
    "LocalRotation": {
      "x": 0.00431122072,
      "y": -0.00127039349,
      "z": -0.005556598,
      "normalized": {
        "x": 0.60324055,
        "y": -0.17775774,
        "z": -0.777497947,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 0.00714676874,
      "sqrMagnitude": 5.10763057E-05
    },
    "Size": {
      "x": 1.12102807,
      "y": 1.6041491,
      "z": 1.19660735,
      "normalized": {
        "x": 0.4887047,
        "y": 0.6993181,
        "z": 0.521653,
        "normalized": {
          "x": 0.488704741,
          "y": 0.6993182,
          "z": 0.521653056,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 2.29387617,
      "sqrMagnitude": 5.26186752
    },
    "ParentUniqueId": 0,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": 0.477163,
      "y": 1.00000036,
      "z": 0.15904662,
      "normalized": {
        "x": 0.426279545,
        "y": 0.8933628,
        "z": 0.1420863,
        "normalized": {
          "x": 0.426279575,
          "y": 0.8933629,
          "z": 0.142086312,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.9999999
      },
      "magnitude": 1.11936641,
      "sqrMagnitude": 1.25298107
    },
    "JointAxis": {
      "x": 0.0,
      "y": 0.0,
      "z": -1.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 120.0,
      "y": 157.0,
      "z": 121.0,
      "normalized": {
        "x": 0.51788646,
        "y": 0.6775681,
        "z": 0.522202134,
        "normalized": {
          "x": 0.5178865,
          "y": 0.677568138,
          "z": 0.5222022,
          "magnitude": 1.00000012,
          "sqrMagnitude": 1.00000024
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 231.711029,
      "sqrMagnitude": 53690.0
    }
  },
  {
    "UniqueId": 2,
    "TypeId": 2,
    "Position": {
      "x": 0.9061055,
      "y": -2.7517755,
      "z": 0.633134842,
      "normalized": {
        "x": 0.3055496,
        "y": -0.927931547,
        "z": 0.213500619,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 2.96549416,
      "sqrMagnitude": 8.794155
    },
    "LocalPosition": {
      "x": 0.9061055,
      "y": -2.7517755,
      "z": 0.633134842,
      "normalized": {
        "x": 0.3055496,
        "y": -0.927931547,
        "z": 0.213500619,
        "magnitude": 1.0,
        "sqrMagnitude": 1.0
      },
      "magnitude": 2.96549416,
      "sqrMagnitude": 8.794155
    },
    "Rotation": {
      "x": 0.00609015161,
      "y": -5.72276349E-06,
      "z": 0.0027681503,
      "normalized": {
        "x": 0.9103718,
        "y": -0.000855453662,
        "z": 0.413790345,
        "normalized": {
          "x": 0.91037184,
          "y": -0.0008554537,
          "z": 0.413790375,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00668974128,
      "sqrMagnitude": 4.47526363E-05
    },
    "LocalRotation": {
      "x": 0.00609015161,
      "y": -5.72276349E-06,
      "z": 0.0027681503,
      "normalized": {
        "x": 0.9103718,
        "y": -0.000855453662,
        "z": 0.413790345,
        "normalized": {
          "x": 0.91037184,
          "y": -0.0008554537,
          "z": 0.413790375,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.00668974128,
      "sqrMagnitude": 4.47526363E-05
    },
    "Size": {
      "x": 1.66683757,
      "y": 2.38518214,
      "z": 1.77921522,
      "normalized": {
        "x": 0.4887047,
        "y": 0.6993182,
        "z": 0.521653056,
        "magnitude": 1.0,
        "sqrMagnitude": 1.00000012
      },
      "magnitude": 3.41072536,
      "sqrMagnitude": 11.6330481
    },
    "ParentUniqueId": 1,
    "JointType": "hinge",
    "JointAnchorPos": {
      "x": 0.5,
      "y": 0.692101061,
      "z": 0.438507676,
      "normalized": {
        "x": 0.5209201,
        "y": 0.7210587,
        "z": 0.456854939,
        "normalized": {
          "x": 0.520920157,
          "y": 0.7210588,
          "z": 0.456854969,
          "magnitude": 1.0,
          "sqrMagnitude": 1.00000012
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 0.959840059,
      "sqrMagnitude": 0.9212929
    },
    "JointAxis": {
      "x": 0.0,
      "y": 0.0,
      "z": -1.0,
      "magnitude": 1.0,
      "sqrMagnitude": 1.0
    },
    "Color": {
      "x": 120.0,
      "y": 157.0,
      "z": 121.0,
      "normalized": {
        "x": 0.51788646,
        "y": 0.6775681,
        "z": 0.522202134,
        "normalized": {
          "x": 0.5178865,
          "y": 0.677568138,
          "z": 0.5222022,
          "magnitude": 1.00000012,
          "sqrMagnitude": 1.00000024
        },
        "magnitude": 0.99999994,
        "sqrMagnitude": 0.99999994
      },
      "magnitude": 231.711029,
      "sqrMagnitude": 53690.0
    }
  }


]


def convert_json_to_blueprint(json_input):
    blueprint = {}
    for item in json_input:
        unique_id = str(item["UniqueId"])
        # if int(unique_id) > 2:
        #     break
        position = (item["Position"]["x"], item["Position"]["z"], item["Position"]["y"])
        rotation = (item["Rotation"]["x"], item["Rotation"]["z"], item["Rotation"]["y"])
        size = (item["Size"]["x"], item["Size"]["z"], item["Size"]["y"])
        color = (item["Color"]["x"], item["Color"]["y"], item["Color"]["z"])
        parent_unique_id = item["ParentUniqueId"]

        if parent_unique_id is not None:
            # parent_position = json_input[parent_unique_id]["Position"]
            parent_item = next((item for item in json_input if item["UniqueId"] == parent_unique_id), None)
            # parent_position = (parent_item["Position"]["x"], parent_item["Position"]["z"], parent_item["Position"]["y"])
            parent_position = (0.0, 0.0, 0.0)
            # swap the 2nd and 3rd elements of the tuple
            # parent_position = (parent_position["x"], parent_position["z"], parent_position["y"])
        
            # adjusted_position = tuple(np.subtract((0.00, 0.00, -5.34), (-0.37, 0.07, -4.02)))[:2] + (-4.02 + 0.42 + 0.00000001,)
            adjusted_position = tuple(np.subtract(np.array(parent_position), np.array(position)))[:2] + (position[2] + size[2] + 0.00000001,) #yz 
            # print(parent_position, position, size, unique_id, parent_unique_id)
            # adjusted_position = tuple(np.subtract(parent_position, position)[:2] + (position[2] + size[2] + 0.00000001,)) #yz swapped, then subtracted from the position of the parent segment, then z value replaced accordingly. TODO: verify that this works.. intuitively it's a bit sus
        else:
            adjusted_position = position

        

        # # Handle parent and subtraction for position if needed
        # if item["ParentUniqueId"] is not None and str(item["ParentUniqueId"]) in blueprint:
        #     parent_position = blueprint[str(item["ParentUniqueId"])]["position"]
        #     parent_size = blueprint[str(item["ParentUniqueId"])]["size"]
        #     # Subtract parent position from current position and adjust Z value
        #     adjusted_position = tuple(np.subtract(position[:2], parent_position[:2])) + (position[2] + parent_size[2] + 0.00000001,)
        # else:
        #     adjusted_position = position
        
        # if int(unique_id) == 1:
        #     adjusted_position = (position[0], position[1], position[2])
        # else:
        #     adjusted_position = position

        # if int(unique_id) != 1:
        #     blueprint[unique_id] = {
        #         'position': adjusted_position,
        #         'rotation': rotation,
        #         'size': size,
        #         'parent_unique_id': item["ParentUniqueId"],
        #         'joint_type': item.get("JointType"),
        #         'joint_anchorpos': None if not item.get("JointAnchorPos") else (
        #             item["JointAnchorPos"]["x"], item["JointAnchorPos"]["z"], item["JointAnchorPos"]["y"]
        #         ),
        #         'joint_axis': None if not item.get("JointAxis") else (
        #             item["JointAxis"]["x"], item["JointAxis"]["z"], item["JointAxis"]["y"]
        #         ),
        #         'color': color
        #     }
        # else:
        #     blueprint[1] = {
        #         'position': tuple(np.subtract((0.00, 0.00, -5.34), (-0.37, 0.07, -4.02)))[:2] + (-4.02 + 0.42 + 0.00000001,),
        #         'rotation': rotation,
        #         'size': size,
        #         'parent_unique_id': item["ParentUniqueId"],
        #         'joint_type': item.get("JointType"),
        #         'joint_anchorpos': None if not item.get("JointAnchorPos") else (
        #             item["JointAnchorPos"]["x"], item["JointAnchorPos"]["z"], item["JointAnchorPos"]["y"]
        #         ),
        #         'joint_axis': None if not item.get("JointAxis") else (
        #             item["JointAxis"]["x"], item["JointAxis"]["z"], item["JointAxis"]["y"]
        #         ),
        #         'color': color
        #     }

    
            


        blueprint[unique_id] = {
            # 'position': position,
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

    return blueprint

blueprint = convert_json_to_blueprint(json_input_3_seg_worm)

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
        action = -20 * math.sin(0.05 * step_counter) * np.ones(action_spec.shape) 
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
