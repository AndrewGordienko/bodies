import numpy as np
import xml.etree.ElementTree as ET

JOINT_RANGE = '-75 75' # This is what we use in our unity evolution simulation.

# NOTE: I believe these two we can tune as hyperparameters.
MOTOR_GEAR = 20000
JOINT_DAMPING = '4.0'

def tuple_to_str(t):
    return ' '.join(map(str, t))

# From this point onwards, we're assuming that everything has already been converted into the mujoco coordinate system (i.e., y and z swapped) before being passed into this.

# Also, joint_type is assumed to be 'hinge' or 'fixed' for all joints for now.
class Segment:
    def __init__(self, unique_id=0, position=(0, 0, 0), rotation=(0, 0, 0), size=(1,1,1), parent_unique_id=0, joint_type='hinge', joint_anchorpos=(0,0,0), joint_axis=(1,0,0), color='0.5 0.5 0.5', creature_id = '0', mujoco_model=None):
        self.unique_id = unique_id
        self.position = position
        self.rotation = rotation
        self.size = size
        self.parent_unique_id = parent_unique_id
        self.joint_type = joint_type
        self.joint_anchorpos = joint_anchorpos
        self.joint_axis = joint_axis
        self.color = color
        self.creature_id = creature_id
        self.name = f'creature_{self.creature_id}_segment_{self.unique_id}'
        self.mujoco_model = mujoco_model

    def to_xml(self, layer):

        # write code to find the element in ET by f'segment_{self.unique_id'
        # segment_parent = ET.find(f".//body[@name='segment_{self.parent_unique_id}']")
        segment_parent = self.mujoco_model.find(f'creature_{self.creature_id}_segment_{self.parent_unique_id}')
        if segment_parent is not None:
            segment = ET.SubElement(segment_parent, 'body', attrib={'name': f'segment_{self.unique_id}', 'pos': tuple_to_str(self.position)})
        else:
            segment = ET.Element('body', attrib={'name': f'segment_{self.unique_id}', 'pos': tuple_to_str(self.position)})
        

        # segment_parent = ET.Element('body', attrib={'name': f'segment_{self.unique_id}', 'pos': tuple_to_str(self.position)})


        ET.SubElement(segment, 'geom', attrib={
            'name': f'{self.name}_geom', 
            'type': 'box', 
            'size': tuple_to_str(self.size), 
            'pos': '0 0 0', 
            # 'pos': tuple_to_str(self.position), 
            'contype': '1', 
            'conaffinity': str(layer),
            'material': self.color  
        })

        if self.parent_unique_id is not None and self.joint_type == 'hinge':
            ET.SubElement(segment, 'joint', attrib={
                'name': f'{self.name}_joint', 
                'type': 'hinge', 
                'pos': tuple_to_str(self.joint_anchorpos), 
                'axis': tuple_to_str(self.joint_axis), 
                'range': JOINT_RANGE, 
                'damping': JOINT_DAMPING, 
                # 'stiffness': '0', 
                # 'armature': '0', 
                # 'solimplimit': '0.99', 
                # 'solreflimit': '0.99', 
                # 'limited': 'true'
            })
        elif self.parent_unique_id is not None and self.joint_type == 'fixed':
            ET.SubElement(segment, 'joint', attrib={
                'name': f'{self.name}_joint', 
                'type': 'fixed', 
                'pos': tuple_to_str(self.joint_anchorpos), 
            })
        # elif self.parent_unique_id is None:
        #     ET.SubElement(segment, 'joint', attrib={
        #         'name': f'{self.name}_joint', 
        #         'type': 'free', 
        #         'pos': '0 0 0', 
        #     })

        # add site for sensors if this is the segment with unique_id 0
        if self.unique_id == 0:
            ET.SubElement(segment, 'site', attrib={
                'name': f'{self.name}_site', 
                'pos': '0 0 0', 
                'type': 'sphere', 
                'size': '0.01'
            })

    
        return segment






