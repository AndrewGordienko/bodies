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
    world_x, world_y = grid_to_world_position(grid_x, grid_position, n_tiles_length, n_tiles_width, tile_size)

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


def grid_to_world_position(grid_x, grid_y, n_tiles_length=9, n_tiles_width=7, tile_size=1):
    # Adjusted grid conversion for the new field size
    world_x = (grid_x - n_tiles_length / 2) * tile_size + tile_size / 2
    world_y = (grid_y - n_tiles_width / 2) * tile_size + tile_size / 2
    return world_x, world_y

def create_player_xml(team, player_id, grid_position):
    """
    Create XML for a player placed based on grid coordinates.
    """
    world_x, world_y = grid_to_world_position(*grid_position)
    pos_str = f'{world_x} {world_y} 0.5'  # Adjust Z position as needed
    player = ET.Element('body', attrib={'name': f'{team}_player_{player_id}', 'pos': pos_str})
    ET.SubElement(player, 'geom', attrib={
        'type': 'box',
        'size': '0.5 0.5 0.5',
        'material': team
    })
    return player

def create_ball_xml(grid_position):
    """
    Create XML for the ball placed based on grid coordinates.
    """
    world_x, world_y = grid_to_world_position(*grid_position)
    pos_str = f'{world_x} {world_y} 0.1'  # Adjust Z position for ball to be on the ground
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

    # Player positions adjusted to the new grid
    worldbody.append(create_player_xml('team_red', 1, (2, 2)))
    worldbody.append(create_player_xml('team_red', 2, (2, 4)))
    worldbody.append(create_player_xml('team_blue', 1, (6, 2)))
    worldbody.append(create_player_xml('team_blue', 2, (6, 4)))
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
