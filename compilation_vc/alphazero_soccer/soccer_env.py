import pygame
import random
from copy import deepcopy

class SoccerEnv:
    def __init__(self):
        pygame.init()

        # Setting parameters
        self.GRID_WIDTH = 9
        self.GRID_HEIGHT = 7
        self.CELL_SIZE = 40
        self.SCREEN_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.SCREEN_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE

        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('Pygame Soccer-like Game')

        self.net_height = 3
        self.net_width = 1  # New width for the goal post
        self.net_top_position = (self.GRID_HEIGHT - self.net_height) // 2

        self.reset_game()
    
    def reset_game(self):
        self.ball_speed = 2.0
        self.ball_possession = random.choice(['A1', 'A2', 'B1', 'B2'])
        self.scores = {'A': 0, 'B': 0}

        self.player_positions = { # 4 3
            'A1': [self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2 - 1], # 2 2
            'A2': [self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2 + 1], # 2 4
            'B1': [self.GRID_WIDTH // 2 + 2, self.GRID_HEIGHT // 2 - 1], # 6 2
            'B2': [self.GRID_WIDTH // 2 + 2, self.GRID_HEIGHT // 2 + 1] # 6 4
        }

        if self.ball_possession == "A1": self.ball_pos_x, self.ball_pos_y = self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2 - 1
        if self.ball_possession == "A2": self.ball_pos_x, self.ball_pos_y = self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2 + 1
        if self.ball_possession == "B1": self.ball_pos_x, self.ball_pos_y = self.GRID_WIDTH // 2 + 2, self.GRID_HEIGHT // 2 - 1
        if self.ball_possession == "B2": self.ball_pos_x, self.ball_pos_y = self.GRID_WIDTH // 2 + 2, self.GRID_HEIGHT // 2 + 1

        self.state = {
            'player_positions': self.player_positions.copy(),
            'ball_pos': [self.ball_pos_x, self.ball_pos_y],
            'ball_possession': self.ball_possession
        }
    
    def check_goal(self):
        if self.net_top_position <= self.ball_pos_y < self.net_top_position + self.net_height:
            if self.ball_pos_x < self.net_width:  
                return 'B'
            elif self.ball_pos_x >= self.GRID_WIDTH - self.net_width:
                return 'A'
        elif self.ball_pos_x == 0 or self.ball_pos_x == self.GRID_WIDTH - 1:
            # Check if the ball is passed over the goal post
            if self.ball_pos_y < self.net_top_position or self.ball_pos_y >= self.net_top_position + self.net_height:
                if self.ball_pos_x == 0:
                    return 'B'
                else:
                    return 'A'
        return None


    def render(self, state):
        self.screen.fill((255, 255, 255))
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
                pygame.draw.rect(self.screen, (200, 200, 200), (x, y, self.CELL_SIZE, self.CELL_SIZE), 1)
        
        pygame.draw.rect(self.screen, (150, 150, 150), (0, self.net_top_position * self.CELL_SIZE, self.net_width * self.CELL_SIZE, self.net_height * self.CELL_SIZE))
        pygame.draw.rect(self.screen, (150, 150, 150), (self.SCREEN_WIDTH - self.net_width * self.CELL_SIZE, self.net_top_position * self.CELL_SIZE, self.net_width * self.CELL_SIZE, self.net_height * self.CELL_SIZE))

        player_colors = {
            'A1': (0, 0, 255), 'A2': (135, 206, 235),
            'B1': (0, 128, 0), 'B2': (50, 205, 50)
        }

        for player, pos in state['player_positions'].items():
            x, y = pos
            pygame.draw.circle(self.screen, player_colors[player], (x*self.CELL_SIZE + self.CELL_SIZE//2, y*self.CELL_SIZE + self.CELL_SIZE//2), self.CELL_SIZE//3)
            
            if [state['ball_pos'][0], state['ball_pos'][1]] == [x, y]:
                state['ball_possession'] = player
        
        ball_x, ball_y = state['ball_pos']
        pygame.draw.circle(self.screen, (255, 0, 0), (ball_x * self.CELL_SIZE + self.CELL_SIZE // 2, ball_y * self.CELL_SIZE + self.CELL_SIZE // 2), self.CELL_SIZE // 4)

        pygame.display.flip()
        return state

    def play_game(self, action_fn, buffer, n_game, num_games=1):
        games_played = 0
        while games_played < num_games:
            self.reset_game()
            state = deepcopy(self.state)
            game_frames = []  # List to store frames for this game
            player_positions = []  # List to store player positions
            ball_positions = []  # List to store ball positions
            actions = []
            reward = 0  # Initialize rewards for both teams
            steps = 0

            prev_ball_pos = None  # To keep track of the ball's previous position
            prev_ball_possession = self.ball_possession  # To keep track of the previous ball possession

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                state = self.render(deepcopy(state))
                game_frames.append(deepcopy(state))  # Save the current frame

                # Store player positions in player_positions list
                player_positions.append(deepcopy(state['player_positions']))

                # Store ball position in ball_positions list
                self.ball_pos_x, self.ball_pos_y = state['ball_pos']
                ball_positions.append(deepcopy(state['ball_pos']))

                if self.check_goal() == 'A':
                    reward = 10
                    break
                elif self.check_goal() == 'B':
                    reward = -10
                    break

                node = action_fn(deepcopy(state))
                state = deepcopy(node.state)
                actions.append(deepcopy(node.action))

                # Check for picking up the ball
                if state['ball_possession'] != prev_ball_possession:
                    if state['ball_possession'] in ['A1', 'A2']:
                        reward += 2  # Team A picked up the ball
                    else:
                        reward -= 2  # Team B picked up the ball

                # Check for shooting the ball (moving it closer to the opponent's goal)
                if prev_ball_pos and state['ball_possession'] in ['A1', 'A2']:
                    if state['ball_pos'][0] > prev_ball_pos[0]:  # Ball moved closer to Team B's goal
                        reward += 0.5
                elif prev_ball_pos and state['ball_possession'] in ['B1', 'B2']:
                    if state['ball_pos'][0] < prev_ball_pos[0]:  # Ball moved closer to Team A's goal
                        reward -= 0.5
                
                prev_ball_possession = state['ball_possession']
                prev_ball_pos = state['ball_pos']

                if steps > 50:
                    # Distance-based reward shaping
                    distance_to_A_goal = state['ball_pos'][0]  # Considering only the x-coordinate
                    distance_to_B_goal = self.GRID_WIDTH - state['ball_pos'][0]
                    normalized_distance_A = distance_to_A_goal / self.GRID_WIDTH
                    normalized_distance_B = distance_to_B_goal / self.GRID_WIDTH
                    reward = normalized_distance_B - normalized_distance_A  # If ball is closer to B's goal, this will be positive, else negative.
                    break
                steps += 1

            # Store frames, player positions, ball positions, and rewards directly in the buffer
            for i, frame in enumerate(game_frames):
                buffer.push({
                    'state': deepcopy(state),  # Store the full state
                    'reward': reward,
                    'actions': actions,
                    'game_id': n_game

                })
            
            n_game += 1

            
            games_played += 1

        return buffer, n_game

