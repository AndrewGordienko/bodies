import numpy as np
import xml.etree.ElementTree as ET
import random

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

class Torso:
    def __init__(self, name="torso", position=(0, 0, 0.75), size=None):
        self.name = name
        self.position = position
        # self.size = size if size else (random.uniform(0.2, 0.5), random.uniform(0.1, 0.2), random.uniform(0.05, 0.1))
        self.size = size if size else (random.uniform(0.2, 0.5), random.uniform(0.1, 5*0.2), random.uniform(0.05, 0.1))

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
    def __init__(self, name, torso_size, upper_size, lower_size, foot_size, position):
        self.name = name
        self.torso_size = torso_size
        # Customizable sizes for leg parts
        self.upper_size = upper_size
        self.lower_size = lower_size
        self.foot_size = foot_size
        self.subparts = 0
        self.position = position

    def to_xml(self):
        # Random edge selection for leg placement
        edge_positions = [
            (0, self.torso_size[1]/2, 0),  # Right side
            (0, -self.torso_size[1]/2, 0),  # Left side
            (self.torso_size[0]/2, 0, 0),  # Front side
            (-self.torso_size[0]/2, 0, 0)  # Back side
        ]
        # position = random.choice(edge_positions)
        position = self.position

        leg = ET.Element('body', attrib={'name': self.name, 'pos': ' '.join(map(str, position))}) 

        # Upper part
        # Adjust the position based on half of its height to place its base at the parent body's attachment point
        upper_pos_z = -self.upper_size[2]
        ET.SubElement(leg, 'geom',
                    attrib={'name': self.name + '_upper_geom',
                            'type': 'box', 
                            'size': ' '.join(map(str, self.upper_size)),
                            'pos': f'0 0 {upper_pos_z}'
                            })

        # ET.SubElement(leg, 'joint', attrib={'name': self.name + '_hip_joint', 'type': 'ball', 'damping': joint_damping['hip']})
        ET.SubElement(leg, 'joint', attrib={'name': self.name + '_hip_joint', 'type': 'hinge', 'axis': '0 1 0', 'damping': joint_damping['hip']})

        # Lower part
        # Position the lower part based on the total length of the upper part to continue from its end
        lower_pos_z = upper_pos_z - self.lower_size[2]
        lower_part = ET.SubElement(leg, 'body', attrib={'name': self.name + '_lower', 'pos': f'0 0 {lower_pos_z}'})
        ET.SubElement(lower_part, 'geom', attrib={'name': self.name + '_lower_geom', 'type': 'box', 'size': ' '.join(map(str, self.lower_size))})

        # Knee joint
        ET.SubElement(lower_part, 'joint', attrib={'name': self.name + '_knee_joint', 'type': 'hinge', 'axis': '0 1 0', 'range': joint_ranges['knee'], 'damping': joint_damping['knee'], 'limited': 'true'})

        # Foot part
        # Position the foot part based on the total length of the upper and lower parts to continue from its end
        foot_pos_z = lower_pos_z - self.foot_size[2]
        foot_part = ET.SubElement(lower_part, 'body', attrib={'name': self.name + '_foot', 'pos': f'0 0 {foot_pos_z}'})
        ET.SubElement(foot_part, 'geom', attrib={'name': self.name + '_foot_geom', 'type': 'box', 'size': ' '.join(map(str, self.foot_size))})

        # Ankle joint
        # ET.SubElement(foot_part, 'joint', attrib={'name': self.name + '_ankle_joint', 'type': 'ball', 'damping': joint_damping['ankle']})
        ET.SubElement(foot_part, 'joint', attrib={'name': self.name + '_ankle_joint', 'type': 'hinge', 'axis': '0 1 0', 'damping': joint_damping['ankle']})

        self.subparts = 1  # upper part
        self.subparts += 1 if self.lower_size[2] > 0 else 0
        self.subparts += 1 if self.foot_size[2] > 0 else 0

        return leg, self.name + '_ankle_joint'
