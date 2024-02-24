import xml.etree.ElementTree as ET

def create_assets_xml():
    asset = ET.Element('asset')

    # Texture for the soccer pitch (for the checkered pattern)
    ET.SubElement(asset, 'texture', attrib={
        'name': 'pitch_texture',
        'type': '2d',
        'builtin': 'checker',
        'rgb1': '0.1 0.8 0.1',  # Light green
        'rgb2': '0.0 0.5 0.0',  # Dark green
        'width': '1024',
        'height': '1024'
    })

    # Material for the soccer pitch using the texture
    ET.SubElement(asset, 'material', attrib={
        'name': 'MatPlane',
        'texture': 'pitch_texture',
        'reflectance': '0.2'
    })

    # Alternate material for the checkered floor pattern
    ET.SubElement(asset, 'material', attrib={
        'name': 'MatCheckeredAlternate',
        'rgba': '0.8 0.9 0.8 1'  # Slightly lighter green
    })

    # Material for the walls
    ET.SubElement(asset, 'material', attrib={
        'name': 'MatWall',
        'rgba': '0.7 0.7 0.7 1'
    })

    # Materials for team players
    teams = ['team_red', 'team_blue']
    team_colors = {'team_red': '1 0 0 1', 'team_blue': '0 0 1 1'}
    for team in teams:
        ET.SubElement(asset, 'material', attrib={
            'name': team,
            'rgba': team_colors[team]
        })

    # Material for the goals
    goal_materials = {'team_red_goal': '1 0.5 0.5 1', 'team_blue_goal': '0.5 0.5 1 1'}
    for goal, rgba in goal_materials.items():
        ET.SubElement(asset, 'material', attrib={
            'name': goal,
            'rgba': rgba
        })

    # Material for the ball
    ET.SubElement(asset, 'material', attrib={
        'name': 'ball_material',
        'rgba': '1 1 1 1'
    })

    return asset
