import xml.etree.ElementTree as ET
import random
from soccer_asset_components import create_assets_xml

tile_size = 1

def create_checkered_floor(n_tiles_length=9, n_tiles_width=7, tile_size=1):
    # Adjusted the field size as requested
    floor_tiles = []
    total_length = n_tiles_length * tile_size
    total_width = n_tiles_width * tile_size

    for i in range(n_tiles_length):
        for j in range(n_tiles_width):
            pos_x = (i + 0.5) * tile_size - total_length / 2
            pos_y = (j + 0.5) * tile_size - total_width / 2
            material_name = 'MatPlane' if (i+j) % 2 == 0 else 'MatCheckeredAlternate'
            floor_tiles.append(ET.Element('geom', attrib={
                'type': 'plane',
                'size': f'{tile_size/2} {tile_size/2} 0.1',
                'pos': f'{pos_x} {pos_y} 0',
                'material': material_name
            }))
    return floor_tiles

def create_walls_xml(total_length, total_width, wall_thickness=0.1, wall_height=0.5):
    half_length = total_length / 2
    half_width = total_width / 2
    walls = [
        ET.Element('geom', attrib={'type': 'box', 'size': f'{total_length/2} {wall_thickness/2} {wall_height}', 'pos': f'0 {half_width + wall_thickness/2} 0', 'material': 'MatWall'}),
        ET.Element('geom', attrib={'type': 'box', 'size': f'{total_length/2} {wall_thickness/2} {wall_height}', 'pos': f'0 {-half_width - wall_thickness/2} 0', 'material': 'MatWall'}),
        ET.Element('geom', attrib={'type': 'box', 'size': f'{wall_thickness/2} {total_width/2} {wall_height}', 'pos': f'{half_length + wall_thickness/2} 0 0', 'material': 'MatWall'}),
        ET.Element('geom', attrib={'type': 'box', 'size': f'{wall_thickness/2} {total_width/2} {wall_height}', 'pos': f'{-half_length - wall_thickness/2} 0 0', 'material': 'MatWall'}),
    ]
    return walls


def create_goal_xml(team, grid_position, n_tiles_length=9, n_tiles_width=7, tile_size=1, goal_size=(0.1, tile_size * 3 / 2, 0.05)):
    """
    Create XML for a goal placed based on grid coordinates.
    Adjusted to place the goal in the center of the back row and flush to the ground.
    """
    # Adjust goal position based on grid coordinates
    if team == 'team_red':
        grid_x = 0  # Red team's goal at the starting row
    else:
        grid_x = n_tiles_length - 1  # Blue team's goal at the opposite end

    # Calculate world position for the goal
    world_x, world_y, world_z = grid_to_world_position(grid_x, grid_position, n_tiles_length, n_tiles_width, tile_size)

    pos_str = f'{world_x} {world_y} {world_z}'

    # Adjust the Y position to center the goal on the back row
    # Since the goal spans 3 tiles, we adjust the grid_y to be in the middle
    goal_area = ET.Element('geom', attrib={
        'name': f'{team}_goal_area',
        'type': 'box',
        'size': ' '.join(map(str, goal_size)),
        'pos': f'{world_x} {world_y} {goal_size[2]}',  # Z position adjusted to make goal flush with ground
        'material': f'{team}_goal'
    })
    return goal_area


def grid_to_world_position(grid_x, grid_y, n_tiles_length=9, n_tiles_width=7, tile_size=1, z_pos=0.25):
    """
    Convert grid coordinates to world coordinates.
    
    Args:
        grid_x (int): Grid X coordinate.
        grid_y (int): Grid Y coordinate.
        n_tiles_length (int): Number of tiles along the length of the field.
        n_tiles_width (int): Number of tiles along the width of the field.
        tile_size (float): Size of each tile.
        z_pos (float): Fixed height (Z coordinate) at which objects should be placed.
        
    Returns:
        tuple: A tuple (world_x, world_y, z_pos) representing world coordinates.
    """
    # Calculate the center of the field
    field_center_x = n_tiles_length * tile_size / 2
    field_center_y = n_tiles_width * tile_size / 2
    
    # Convert grid coordinates to world coordinates
    world_x = grid_x * tile_size - field_center_x + tile_size / 2
    world_y = grid_y * tile_size - field_center_y + tile_size / 2
    
    return (world_x, world_y, z_pos)

def create_ball_xml(grid_position):
    """
    Create XML for the ball placed based on grid coordinates.
    """
    world_x, world_y, world_z = grid_to_world_position(4, 3, 9, 7, tile_size)

    pos_str = f'{world_x} {world_y} {world_z}'

    ball = ET.Element('body', attrib={'name': 'ball', 'pos': pos_str})
    ET.SubElement(ball, 'geom', attrib={
        'type': 'sphere',
        'size': '0.25',
        'material': 'ball_material'
    })
    return ball


def create_soccer_environment(n_tiles_length=9, n_tiles_width=7, tile_size=1):
    # Adjusted the field dimensions as requested
    total_length = n_tiles_length * tile_size
    total_width = n_tiles_width * tile_size

    mujoco = ET.Element('mujoco')
    assets = create_assets_xml()
    mujoco.append(assets)

    worldbody = ET.SubElement(mujoco, 'worldbody')

    floor_tiles = create_checkered_floor(n_tiles_length, n_tiles_width, tile_size)
    for tile in floor_tiles:
        worldbody.append(tile)
    
    walls = create_walls_xml(total_length, total_width)
    for wall in walls:
        worldbody.append(wall)

    player_names = [
        "team_red_player_1", "team_red_player_2",
        "team_blue_player_1", "team_blue_player_2"
    ]

    color_materials = {
        'red': '1 0 0 1', 
        'green': '0 1 0 1', 
        'blue': '0 0 1 1',
        'yellow': '1 1 0 1', 
        'purple': '0.5 0 0.5 1', 
        'orange': '1 0.5 0 1',
        'pink': '1 0.7 0.7 1', 
        'grey': '0.5 0.5 0.5 1', 
        'brown': '0.6 0.3 0 1'
    }

    colors = ['team_red', 'team_red', 'team_blue', 'team_blue']
    actuator = ET.SubElement(mujoco, 'actuator')
    creature_grid_positions = [(2, 2), (2, 4), (6, 2), (6, 4)]

    creature_leg_info = {}
    for creature_id in range(4):
        layer = creature_id + 1
        color = colors[creature_id % len(colors)]

        grid_x, grid_y = creature_grid_positions[creature_id]
        world_x, world_y, world_z = grid_to_world_position(grid_x, grid_y, n_tiles_length, n_tiles_width, tile_size)
        initial_position = [world_x, world_y, 1]  # Z position set to 1 for all creatures
        
        # Create torso with adjusted position
        torso_obj = Torso(name=player_names[creature_id], position=initial_position)
        torso_xml = torso_obj.to_xml(layer, colors[creature_id % len(colors)])
        worldbody.append(torso_xml)


        num_legs = random.randint(1, 4)
        leg_size = 0.04
        leg_info = []

        for i in range(num_legs):
            leg_name = f"leg_{creature_id}_{i+1}"

            # Create Leg object with random edge placement
            leg_obj = Leg(leg_name, torso_obj.size, leg_size)
            leg_xml, foot_joint_name = leg_obj.to_xml()
            torso_xml.append(leg_xml)

            # Add motors for each joint
            ET.SubElement(actuator, 'motor', attrib={
                'name': f'{leg_name}_hip_motor',
                'joint': f'{leg_name}_hip_joint',
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'gear': str(motor_gears['hip'])
            })

            # Add motors for knee and ankle if they exist
            if 'knee_joint' in foot_joint_name:
                ET.SubElement(actuator, 'motor', attrib={
                    'name': f'{leg_name}_knee_motor',
                    'joint': f'{leg_name}_knee_joint',
                    'ctrllimited': 'true',
                    'ctrlrange': '-1 1',
                    'gear': str(motor_gears['knee'])
                })

            if 'ankle_joint' in foot_joint_name:
                ET.SubElement(actuator, 'motor', attrib={
                    'name': f'{foot_joint_name}_motor',
                    'joint': foot_joint_name,
                    'ctrllimited': 'true',
                    'ctrlrange': '-1 1',
                    'gear': str(motor_gears['ankle'])
                })
            
            leg_info.append(leg_obj.subparts)
        
        creature_leg_info[creature_id] = leg_info

    # Add sensors
    sensors = ET.SubElement(mujoco, 'sensor')
    for creature_id in range(4):
        torso_name = player_names[creature_id]
        ET.SubElement(sensors, 'accelerometer', attrib={'name': f'{torso_name}_accel', 'site': f'{torso_name}_site'})
        ET.SubElement(sensors, 'gyro', attrib={'name': f'{torso_name}_gyro', 'site': f'{torso_name}_site'})

    worldbody.append(create_ball_xml((4, 3)))  # Centered ball

    # Adjusting goal positions for the new field dimensions
    # Goals are centered on the back row
    goal_grid_y = n_tiles_width // 2  # Use integer division for center tile
    red_goal_position = (0, goal_grid_y)  # For red team at the north end
    blue_goal_position = (n_tiles_length - 1, goal_grid_y)  # For blue team at the south end

    # Extract grid_x and grid_y from the tuple for red and blue goals
    red_goal_grid_x, red_goal_grid_y = red_goal_position
    blue_goal_grid_x, blue_goal_grid_y = blue_goal_position

    # Now pass grid_x and grid_y separately to create_goal_xml
    worldbody.append(create_goal_xml('team_red', red_goal_grid_y, n_tiles_length, n_tiles_width, tile_size, goal_size=(0.1, tile_size * 3 / 2, 0.05)))
    worldbody.append(create_goal_xml('team_blue', blue_goal_grid_y, n_tiles_length, n_tiles_width, tile_size, goal_size=(0.1, tile_size * 3 / 2, 0.05)))


    xml_str = ET.tostring(mujoco, encoding='unicode')
    return xml_str

import numpy as np
import xml.etree.ElementTree as ET
import random

joint_ranges = {
    'hip': '-90 90',
    'knee': '-90 90',
    'ankle': '-50 50'  # New ankle joint range
}
motor_gears = {
    'hip': 200,
    'knee': 200,
    'ankle': 200  # New gear for ankle motor
}

# Lower damping values for more fluid movement
joint_damping = {
    'hip': '2.0',
    'knee': '4.0',
    'ankle': '6.0'  # New damping value for ankle joint
}

class Torso:
    def __init__(self, name="torso", position=(0, 0, 0.75), size=None):
        self.name = name
        self.position = position
        self.size = size if size else (random.uniform(0.2, 0.5), random.uniform(0.1, 0.2), random.uniform(0.05, 0.1))

    def to_xml(self, layer, color):
        torso = ET.Element('body', attrib={'name': self.name, 'pos': ' '.join(map(str, self.position))})
        ET.SubElement(torso, 'geom', attrib={
            'name': f'torso_geom_{self.name}', 
            'type': 'box', 
            'size': ' '.join(map(str, self.size)), 
            'pos': '0 0 0', 
            'contype': '1', 
            'conaffinity': str(layer),
            'material': color  # Assign unique color
        })
        
        ET.SubElement(torso, 'joint', attrib={
            'name': f'{self.name}_root', 
            'type': 'free', 
            'armature': '0', 
            'damping': '0', 
            'limited': 'false'
        })
        ET.SubElement(torso, 'site', attrib={
            'name': f'{self.name}_site', 
            'pos': '0 0 0', 
            'type': 'sphere', 
            'size': '0.01'
        })

        return torso


class Leg:
    def __init__(self, name, torso_size, size):
        self.name = name
        self.torso_size = torso_size
        self.size = size
        self.subparts = 0

    def to_xml(self):
        # Random edge selection for leg placement
        edge_positions = [
            (0, self.torso_size[1]/2, 0),  # Right side
            (0, -self.torso_size[1]/2, 0),  # Left side
            (self.torso_size[0]/2, 0, 0),  # Front side
            (-self.torso_size[0]/2, 0, 0)  # Back side
        ]
        position = random.choice(edge_positions)

        leg = ET.Element('body', attrib={'name': self.name, 'pos': ' '.join(map(str, position))})

        # Random lengths for each part with a small overlap
        upper_length = np.random.uniform(0.1, 0.2)
        lower_length = np.random.uniform(0.1, 0.2)
        foot_length = np.random.uniform(0.1, 0.2)

        # Upper part
        upper_fromto = [0.0, 0.0, 0.0, upper_length, 0.0, 0.0]
        ET.SubElement(leg, 'geom', attrib={'name': self.name + '_upper_geom', 'type': 'capsule', 'fromto': ' '.join(map(str, upper_fromto)), 'size': str(self.size)})
        ET.SubElement(leg, 'joint', attrib={'name': self.name + '_hip_joint', 'type': 'ball', 'damping': joint_damping['hip']})

        # Lower part
        lower_fromto = [upper_length, 0.0, 0.0, upper_length + lower_length, 0.0, 0.0]
        lower_part = ET.SubElement(leg, 'body', attrib={'name': self.name + '_lower', 'pos': ' '.join(map(str, [upper_length, 0.0, 0.0]))})
        ET.SubElement(lower_part, 'geom', attrib={'name': self.name + '_lower_geom', 'type': 'capsule', 'fromto': ' '.join(map(str, lower_fromto)), 'size': str(self.size)})

        # Knee joint
        ET.SubElement(lower_part, 'joint', attrib={'name': self.name + '_knee_joint', 'type': 'hinge', 'axis': '0 1 0', 'range': joint_ranges['knee'], 'damping': joint_damping['knee'], 'limited': 'true'})

        # Foot part
        foot_fromto = [upper_length + lower_length, 0.0, 0.0, upper_length + lower_length + foot_length, 0.0, 0.0]
        foot_part = ET.SubElement(lower_part, 'body', attrib={'name': self.name + '_foot', 'pos': ' '.join(map(str, [upper_length + lower_length, 0.0, 0.0]))})
        ET.SubElement(foot_part, 'geom', attrib={'name': self.name + '_foot_geom', 'type': 'cylinder', 'fromto': ' '.join(map(str, foot_fromto)), 'size': str(self.size)})
        ET.SubElement(foot_part, 'joint', attrib={'name': self.name + '_ankle_joint', 'type': 'ball', 'damping': joint_damping['ankle']})

        self.subparts = 1  # upper part
        self.subparts += 1 if lower_length > 0 else 0
        self.subparts += 1 if foot_length > 0 else 0

        return leg, self.name + '_ankle_joint',
