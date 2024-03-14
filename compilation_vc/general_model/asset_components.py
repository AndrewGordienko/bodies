import numpy as np
import xml.etree.ElementTree as ET
import random
from body_components import Torso, Leg
import math

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

def create_assets_xml():
    assets = ET.Element('asset')

    # Add a checkered texture for the floor
    ET.SubElement(assets, 'texture', attrib={
        'name': 'checkered',
        'type': '2d',
        'builtin': 'checker',
        'rgb1': '0.2 0.3 0.4',  # Color 1 of the checker pattern
        'rgb2': '0.9 0.9 0.9',  # Color 2 of the checker pattern
        'width': '512',         # Texture width
        'height': '512'         # Texture height
    })

    # Add a material that uses the checkered texture
    ET.SubElement(assets, 'material', attrib={
        'name': 'MatCheckered',
        'texture': 'checkered',
        'reflectance': '0.5'
    })

    # Material for the plane (floor)
    ET.SubElement(assets, 'material', attrib={
        'name': 'MatPlane',
        'reflectance': '0.5',
        'shininess': '1',
        'specular': '1'
    })

    # Define materials for the colors used for creatures and flags
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

    for name, rgba in color_materials.items():
        ET.SubElement(assets, 'material', attrib={'name': name, 'rgba': rgba})

    return assets

def create_floor_xml(size=(10, 10, 0.1), friction=(1.0, 0.005, 0.0001)):
    return ET.Element('geom', attrib={
        'name': 'floor', 
        'type': 'plane', 
        'size': ' '.join(map(str, size)), 
        'pos': '0 0 0', 
        'material': 'MatCheckered',
        'friction': ' '.join(map(str, friction))
        })

def create_flag_xml(flag_id, layer, color, center_position, dynamic_radius):
    # Use dynamic_radius instead of a static radius
    angle = random.uniform(0, 2 * math.pi)
    flag_x = center_position[0] + dynamic_radius * math.cos(angle)
    flag_y = center_position[1] + dynamic_radius * math.sin(angle)
    flag_z = 0
    flag_position = (flag_x, flag_y, flag_z)
    flag_size = (0.05, 0.05, 0.5)
    return ET.Element('geom', attrib={
        'name': f'flag_{flag_id}', 
        'type': 'box', 
        'size': ' '.join(map(str, flag_size)), 
        'pos': ' '.join(map(str, flag_position)), 
        'material': color,
        'contype': '1', 
        'conaffinity': str(layer)
    })

def create_ball_with_joint_xml(name, pos, size):
    # Create the main body element
    body = ET.Element('body', attrib={
        'name': name,
        'pos': ' '.join(map(str, pos))
    })

    # Create the geom element as a child of the body
    geom = ET.SubElement(body, 'geom', attrib={
        'type': 'sphere',
        'size': ' '.join(map(str, size)),
        'material': 'MatPlane'
    })

    # Create a free joint element as a child of the body
    joint = ET.SubElement(body, 'freejoint', attrib={
        'name': f'{name}_joint'
    })

    return body


def create_wall_xml(name, pos, size):
    return ET.Element('geom', attrib={
        'name': name,
        'type': 'box',
        'pos': ' '.join(map(str, pos)),
        'size': ' '.join(map(str, size)),
        'material': 'MatPlane'
    })

def create_ant_model(flag_radius):
    num_creatures=9
    # Ensure the function is optimized for 9 creatures
    assert num_creatures == 9, "This setup is optimized for 9 creatures."
    
    mujoco_model = ET.Element('mujoco')
    mujoco_model.append(create_assets_xml())
    worldbody = ET.SubElement(mujoco_model, 'worldbody')
    world_size = (10, 10, 0.1)  # Define the world size
    world_friction = (1.0, 0.005, 0.0001)
    worldbody.append(create_floor_xml(size=world_size, friction=world_friction))

    # gravity
    option = ET.SubElement(mujoco_model, 'option')
    option.set('gravity', '0 0 -9.81')  # add gravity in -z direction


    actuator = ET.SubElement(mujoco_model, 'actuator')

    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'grey', 'brown']

    creature_leg_info = {}

    # Define positions to maximize the spread, using the entire floor
    edge_offset = 6.5  # Slightly less than half the floor size to keep creatures fully within bounds
    max_spread_positions = [
        (-edge_offset, -edge_offset, 0.75),  # Bottom-left corner
        (0, -edge_offset, 0.75),             # Bottom edge center
        (edge_offset, -edge_offset, 0.75),   # Bottom-right corner
        (-edge_offset, 0, 0.75),             # Left edge center
        (0, 0, 0.75),                        # Center of the floor
        (edge_offset, 0, 0.75),              # Right edge center
        (-edge_offset, edge_offset, 0.75),   # Top-left corner
        (0, edge_offset, 0.75),              # Top edge center
        (edge_offset, edge_offset, 0.75)     # Top-right corner
    ]

    for creature_id in range(num_creatures):
        layer = creature_id + 1
        color = colors[creature_id % len(colors)]
        initial_position = max_spread_positions[creature_id]
        # flag_radius = 3.5  # Adjust the radius as needed 3.5 is ideal flag radius

        torso_obj = Torso(name=f'torso_{creature_id}', position=initial_position)
        torso_xml = torso_obj.to_xml(layer, color)
        worldbody.append(torso_xml)

        worldbody.append(create_flag_xml(creature_id, layer, color, initial_position, flag_radius))

        # offset initial_position
        initial_position = (initial_position[0] + 1.5, initial_position[1], initial_position[2])

        # worldbody.append(create_ball_with_joint_xml(f'ball_{creature_id}', initial_position, (0.5, 0.5, 0.5)))
        worldbody.append(create_ball_with_joint_xml(f'ball_{creature_id}', initial_position, (0.1, 0.1, 0.1)))

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
    sensors = ET.SubElement(mujoco_model, 'sensor')
    for creature_id in range(num_creatures):
        torso_name = f'torso_{creature_id}'
        ET.SubElement(sensors, 'accelerometer', attrib={'name': f'{torso_name}_accel', 'site': f'{torso_name}_site'})
        ET.SubElement(sensors, 'gyro', attrib={'name': f'{torso_name}_gyro', 'site': f'{torso_name}_site'})

    wall_thickness = 0.1
    wall_height = 0.5
    floor_size = 10
    half_floor_size = floor_size / 2
    wall_length = floor_size + wall_thickness  # Extend wall length to span outside the floor area

    # Calculate positions for the walls so they encircle the floor
    walls = [
        # North wall
        ('north_wall', (0, half_floor_size + wall_thickness / 2 + 5, wall_height / 2), (floor_size / 2 + wall_thickness / 2 + 5, wall_thickness / 2, wall_height / 2)),
        # South wall
        ('south_wall', (0, -half_floor_size - wall_thickness / 2 - 5, wall_height / 2), (floor_size / 2 + wall_thickness / 2 + 5, wall_thickness / 2, wall_height / 2)),
        # East wall
        ('east_wall', (half_floor_size + wall_thickness / 2 + 5, 0, wall_height / 2), (wall_thickness / 2, floor_size / 2 + wall_thickness / 2 + 5, wall_height / 2)),
        # West wall
        ('west_wall', (-half_floor_size - wall_thickness / 2 - 5, 0, wall_height / 2), (wall_thickness / 2, floor_size / 2 + wall_thickness / 2 + 5, wall_height / 2))
    ]

    for name, pos, size in walls:
        worldbody.append(create_wall_xml(name, pos, size))
        
    # Rest of your function remains unchanged
    xml_string = ET.tostring(mujoco_model, encoding='unicode')
    return xml_string, creature_leg_info

    xml_string = ET.tostring(mujoco_model, encoding='unicode')
    return xml_string, creature_leg_info
