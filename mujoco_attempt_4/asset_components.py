import numpy as np
import xml.etree.ElementTree as ET
import random
from body_components import Torso, Leg

# joint_ranges = {
#     'hip': '-90 90',
#     'knee': '-90 90',
#     'ankle': '-50 50'  # New ankle joint range
# }
joint_ranges = {
    'hip': '-75 75',
    'knee': '-75 75',
    'ankle': '-75 75'  # New ankle joint range
}

# NOTE: I believe this we can tune as a hyperparameter
motor_gears = {
    'hip': 200,
    'knee': 200,
    'ankle': 200  # New gear for ankle motor
}

# NOTE: I believe this we can tune as a hyperparameter
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
    # color_materials = {
    #     'red': '1 0 0 1', 
    #     'green': '0 1 0 1', 
    #     'blue': '0 0 1 1',
    #     'yellow': '1 1 0 1', 
    #     'purple': '0.5 0 0.5 1', 
    #     'orange': '1 0.5 0 1',
    #     'pink': '1 0.7 0.7 1', 
    #     'grey': '0.5 0.5 0.5 1', 
    #     'brown': '0.6 0.3 0 1'
    # }
    alpha = '0.2'

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

def create_floor_xml(size=(10, 10, 0.1)):
    return ET.Element('geom', attrib={'name': 'floor', 'type': 'plane', 'size': ' '.join(map(str, size)), 'pos': '0 0 0', 'material': 'MatCheckered'})

def create_flag_xml(flag_id, layer, color, floor_size=(10, 10, 0.1)):
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
        'material': color,  # Assign the color material
        'contype': '1', 
        'conaffinity': str(layer)
    })

def create_ant_model(num_creatures=9):
    mujoco_model = ET.Element('mujoco')
    mujoco_model.append(create_assets_xml())
    worldbody = ET.SubElement(mujoco_model, 'worldbody')
    worldbody.append(create_floor_xml(size=(10, 10, 0.1)))

    actuator = ET.SubElement(mujoco_model, 'actuator')

    # Define a list of colors for creatures and flags
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'grey', 'brown']  # Add more colors if needed

    creature_leg_info = {}  # Dictionary to store leg and subpart info

    for creature_id in range(num_creatures):
        layer = creature_id + 1
        color = colors[creature_id % len(colors)]

        # Adjust the initial position to spread out the creatures
        initial_position = (creature_id - num_creatures / 2, 0, 0.75)
        
        torso_obj = Torso(name=f'torso_{creature_id}', position=initial_position)
        torso_xml = torso_obj.to_xml(layer, color)
        worldbody.append(torso_xml)

        # Create a flag with the same color as the torso
        worldbody.append(create_flag_xml(creature_id, layer, color))

        # num_legs = random.randint(1, 4)
        num_legs = 1
        # upper_size = (0.04, 0.04, 0.04*4)
        # lower_size = (0.04, 0.04, 0.04*4)
        # foot_size = (0.04, 0.04, 0.04*4)
        upper_size = (0.04, 0.04, 0.04)
        lower_size = (0.04, 0.04, 0.04)
        foot_size = (0.04, 0.04, 0.04)
        # upper_size = (0.04*4, 0.04, 0.04)
        # lower_size = (0.04*4, 0.04, 0.04)
        # foot_size = (0.04*4, 0.04, 0.04)
        position = (0, -torso_obj.size[1]/2 - 0.04, 0)

        leg_info = []

        for i in range(num_legs):
            leg_name = f"leg_{creature_id}_{i+1}"

            # Create Leg object with random edge placement
            leg_obj = Leg(leg_name, torso_obj.size, upper_size, lower_size, foot_size, position)
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

    xml_string = ET.tostring(mujoco_model, encoding='unicode')
    return xml_string, creature_leg_info
