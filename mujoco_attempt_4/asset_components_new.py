import numpy as np
import xml.etree.ElementTree as ET
import random
from body_components_new import Segment

JOINT_RANGE = '-75.0 75.0' # This is what we use in our unity evolution simulation.

# NOTE: I believe these two we can tune as hyperparameters.
MOTOR_GEAR = 20000.0
JOINT_DAMPING = '4.0'

FLOOR_SIZE = (10.0, 10.0, 0.1)

CREATURE_SPACING = 2.0
CREATURE_SPAWN_HEIGHT = 10.0

def create_creature_blueprint():
    pass

def swap_yz(t):
    return (t[0], t[2], t[1])

def tuple_to_str(t):
    return ' '.join(map(str, t))

def add_collision_exclusions(mujoco_model, exclusions):
    contact = ET.SubElement(mujoco_model, 'contact')
    for ex in exclusions:
        ET.SubElement(contact, 'exclude', attrib={'body1': ex[0], 'body2': ex[1]})

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

    # Add a material for the plane (floor)
    ET.SubElement(assets, 'material', attrib={
        'name': 'MatPlane',
        'reflectance': '0.5',
        'shininess': '1',
        'specular': '1'
    })

    # alpha = '0.2'
    alpha = '1.0'

    color_materials = {
        'red': f'1 0 0 {alpha}', 
        'green': f'0 1 0 {alpha}', 
        'blue': f'0 0 1 {alpha}',
        'yellow': f'1 1 0 {alpha}', 
        'purple': f'0.5 0 0.5 {alpha}', 
        'orange': f'1 0.5 0 {alpha}',
        'pink': f'1 0.7 0.7 {alpha}', 
        'grey': f'0.5 0.5 0.5 {alpha}', 
        'brown': f'0.6 0.3 0 {alpha}'
    }

    for name, rgba in color_materials.items():
        ET.SubElement(assets, 'material', attrib={'name': name, 'rgba': rgba})
    
    return assets

def create_floor_xml(size=FLOOR_SIZE):
    return ET.Element('geom', attrib={'name': 'floor', 'type': 'plane', 'size': tuple_to_str(size), 'pos': '0 0 0', 'material': 'MatCheckered'})

def create_flag_xml(flag_id, layer, color, floor_size=FLOOR_SIZE):
    flag_x = random.uniform(-floor_size[0]/2, floor_size[0]/2)
    flag_y = random.uniform(-floor_size[1]/2, floor_size[1]/2)
    flag_z = 0  # On the floor
    flag_position = (flag_x, flag_y, flag_z)

    flag_size = (0.05, 0.05, 0.5)  # Cube size
    return ET.Element('geom', attrib={
        'name': f'flag_{flag_id}', 
        'type': 'box', 
        'size': ' '.join(map(str, flag_size)), 
        'pos': ' '.join(map(str, flag_position)), 
        'material': color, 
        'contype': '1', 
        'conaffinity': str(layer)
    })

def create_flags_and_creatures(num_creatures=9, blueprint={}):
    mujoco_model = ET.Element('mujoco')
    mujoco_model.append(create_assets_xml())
    worldbody = ET.SubElement(mujoco_model, 'worldbody')
    worldbody.append(create_floor_xml(size=FLOOR_SIZE))

    actuator = ET.SubElement(mujoco_model, 'actuator')

    # creature_leg_info = {}  # Dictionary to store leg and subpart info
    
    # Define a list of colors for flags and creatures
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'grey', 'brown']  # Add more colors if needed

    exclusions = []
    for creature_id in range(num_creatures):
        layer = creature_id + 1
        color = colors[creature_id % len(colors)]

        worldbody.append(create_flag_xml(creature_id, layer, color))

        # Adjust the initial position to spread out the creatures
        initial_position = (creature_id - num_creatures / CREATURE_SPACING, 0, CREATURE_SPAWN_HEIGHT)

        creature_xml = create_creature_xml(creature_id, layer, color, initial_position, blueprint, actuator)
        worldbody.append(creature_xml)

        for segment_id in range(len(blueprint)):
            for i in range(segment_id + 1, len(blueprint)):
                print(f'creature_{creature_id}_segment_{segment_id}', f'creature_{creature_id}_segment_{i}')
                exclusions.append((f'creature_{creature_id}_segment_{segment_id}', f'creature_{creature_id}_segment_{i}'))
            # exclusions.append((f'creature_{creature_id}_segment_{segment_id}', f'creature_{creature_id}_segment_{segment_id + 1}'))

    # Add sensors
    sensors = ET.SubElement(mujoco_model, 'sensor')
    for creature_id in range(num_creatures):
        creature_name = f'creature_{creature_id}'
        ET.SubElement(sensors, 'accelerometer', attrib={'name': f'{creature_name}_accel', 'site': f'creature_{creature_id}_segment_0_site'})
        ET.SubElement(sensors, 'gyro', attrib={'name': f'{creature_name}_gyro', 'site': f'creature_{creature_id}_segment_0_site'})
        
    xml_string = ET.tostring(mujoco_model, encoding='unicode')
    return xml_string, {}
    # return xml_string

def create_creature_xml(creature_id, layer, color, initial_position, blueprint, actuator):
    creature = ET.Element('body', attrib={'name': f'creature_{creature_id}', 'pos': tuple_to_str(initial_position)})

    for segment_id, segment_info in blueprint.items():
        
        segment = Segment(
            unique_id=int(segment_id),
            position=segment_info['position'],
            rotation=segment_info['rotation'],
            size=segment_info['size'],
            parent_unique_id=segment_info['parent_unique_id'],
            joint_type=segment_info['joint_type'],
            joint_anchorpos=segment_info['joint_anchorpos'],
            joint_axis=segment_info['joint_axis'],
            color=color,
            creature_id=creature_id
        )
        creature.append(segment.to_xml(layer))

        if segment.joint_type == 'hinge':
            # Add motors for each joint
            ET.SubElement(actuator, 'motor', attrib={
                # 'name': f'{leg_name}_hip_motor',
                # 'joint': f'{leg_name}_hip_joint',
                'name': f'{segment.name}_motor',
                'joint': f'{segment.name}_joint',
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'gear': str(MOTOR_GEAR),
            })

    return creature








