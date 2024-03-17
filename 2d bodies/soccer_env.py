import pygame
import numpy as np
import random
import math

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

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

        self.net_height = 3 * self.CELL_SIZE
        self.net_top_position = (self.SCREEN_HEIGHT - self.net_height) // 2

        self.reset_game()

    def reset_game(self):
        self.ball_pos_x, self.ball_pos_y = self.GRID_WIDTH // 2 * self.CELL_SIZE, self.GRID_HEIGHT // 2 * self.CELL_SIZE
        self.ball_target_x = self.ball_pos_x
        self.ball_target_y = self.ball_pos_y
        self.ball_speed = 2.0
        self.ball_possession = random.choice(['A1', 'A2', 'B1', 'B2'])
        self.scores = {'A': 0, 'B': 0}

        self.player_positions = {
            'A1': [self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2 - 1],
            'A2': [self.GRID_WIDTH // 2 - 2, self.GRID_HEIGHT // 2 + 1],
            'B1': [self.GRID_WIDTH // 2 + 2, self.GRID_HEIGHT // 2 - 1],
            'B2': [self.GRID_WIDTH // 2 + 2, self.GRID_HEIGHT // 2 + 1]
        }
        
        self.teamA_distances_in_episode = []
        self.teamB_distances_in_episode = []

        self.game_history = []

    def check_goal(self):
        if self.net_top_position <= self.ball_pos_y <= self.net_top_position + self.net_height:
            if self.ball_pos_x <= self.CELL_SIZE:  
                return 'B'
            elif self.ball_pos_x >= self.SCREEN_WIDTH - self.CELL_SIZE:  
                return 'A'
        return None

    def step(self, actions):
        for player, action in actions.items():
            #print("Position of player", player, ":", self.player_positions[player])
            if player == self.ball_possession and "SHOOT" in action:
                self.ball_possession = None
                if action == 'SHOOT_LEFT':
                    self.ball_target_x = max(0, self.ball_pos_x - self.ball_speed)
                elif action == 'SHOOT_RIGHT':
                    self.ball_target_x = min(self.SCREEN_WIDTH, self.ball_pos_x + self.ball_speed)
                elif action == 'SHOOT_UP':
                    self.ball_target_y = max(0, self.ball_pos_y - self.ball_speed)
                elif action == 'SHOOT_DOWN':
                    self.ball_target_y = min(self.SCREEN_HEIGHT, self.ball_pos_y + self.ball_speed)
            else:
                if action == 'MOVE_LEFT' and self.player_positions[player][0] > 0:
                    self.player_positions[player][0] -= 1
                elif action == 'MOVE_RIGHT' and self.player_positions[player][0] < self.GRID_WIDTH - 1:
                    self.player_positions[player][0] += 1
                elif action == 'MOVE_UP' and self.player_positions[player][1] > 0:
                    self.player_positions[player][1] -= 1
                elif action == 'MOVE_DOWN' and self.player_positions[player][1] < self.GRID_HEIGHT - 1:
                    self.player_positions[player][1] += 1
                elif action == 'PICK' and self.is_adjacent_to_ball(player):
                    self.ball_possession = player

        if self.ball_possession is None:
            dx = self.ball_target_x - self.ball_pos_x
            dy = self.ball_target_y - self.ball_pos_y
            dist = max(abs(dx), abs(dy))
            if dist > 0:
                # Ensure ball stays within screen bounds
                new_ball_x = self.ball_pos_x + self.ball_speed * dx / dist
                new_ball_y = self.ball_pos_y + self.ball_speed * dy / dist
                self.ball_pos_x = max(0, min(self.SCREEN_WIDTH, new_ball_x))
                self.ball_pos_y = max(0, min(self.SCREEN_HEIGHT, new_ball_y))

        goal_scored = False
        scoring_team = self.check_goal()
        if scoring_team:
            self.scores[scoring_team] += 1
            print(f"Team {scoring_team} scored! Current Scores: A - {self.scores['A']} : B - {self.scores['B']}")
            self.ball_pos_x, self.ball_pos_y = self.GRID_WIDTH // 2 * self.CELL_SIZE, self.GRID_HEIGHT // 2 * self.CELL_SIZE

            self.ball_target_x = self.ball_pos_x
            self.ball_target_y = self.ball_pos_y
            self.ball_possession = random.choice(['A1', 'A2', 'B1', 'B2'])

            goal_scored = True
        
        distA = np.linalg.norm((np.array(self.player_positions["A2"])-np.array(self.player_positions["A1"])))
        distB = np.linalg.norm((np.array(self.player_positions["B2"])-np.array(self.player_positions["B1"])))
        
        self.teamA_distances_in_episode.append(distA)
        self.teamB_distances_in_episode.append(distB)
        
        return goal_scored


    def is_adjacent_to_ball(self, player):
        player_x, player_y = self.player_positions[player]
        ball_x = self.ball_pos_x // self.CELL_SIZE
        ball_y = self.ball_pos_y // self.CELL_SIZE
        return abs(player_x - ball_x) <= 1 and abs(player_y - ball_y) <= 1

    def record_step(self, actions):
        self.game_history.append({
            'player_positions': self.player_positions.copy(),
            'ball_pos': (self.ball_pos_x, self.ball_pos_y),
            'ball_possession': self.ball_possession,
            'actions': actions
        })

    def record_to_buffer(self, buffer):
        winner = self.check_goal()
        reward = 1 if winner == 'A' else -1 if winner == 'B' else 0.5
        for step in reversed(self.game_history):
            buffer.push((step['player_positions'], step['ball_pos'], step['ball_possession'], step['actions'], reward))
            reward *= -0.9

    def play_game(self, get_actions_fn, buffer, num_games):
        games_played = 0

        while games_played < num_games:
            goal_scored = False

            while not goal_scored:
                actions = get_actions_fn(self)
                self.record_step(actions)
                goal_scored = self.step(actions)
                #self.render()
                
                if goal_scored:
                    print("End of this game")
                    print("Team A avg distance:", sum(self.teamA_distances_in_episode)/len(self.teamA_distances_in_episode))
                    print("Team B avg distance:", sum(self.teamB_distances_in_episode)/len(self.teamB_distances_in_episode))
                    break

            games_played += 1
            self.record_to_buffer(buffer)






    def render(self):
        self.screen.fill((255, 255, 255))
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
                pygame.draw.rect(self.screen, (200, 200, 200), (x, y, self.CELL_SIZE, self.CELL_SIZE), 1)

        player_colors = {
            'A1': (0, 0, 255), 'A2': (135, 206, 235),
            'B1': (0, 128, 0), 'B2': (50, 205, 50)
        }

        for player, pos in self.player_positions.items():
            x, y = pos
            pygame.draw.circle(self.screen, player_colors[player], (x*self.CELL_SIZE + self.CELL_SIZE//2, y*self.CELL_SIZE + self.CELL_SIZE//2), self.CELL_SIZE//3)

        if self.ball_possession:
            ball_x, ball_y = self.player_positions[self.ball_possession]
            pygame.draw.circle(self.screen, (255, 0, 0), (ball_x*self.CELL_SIZE + self.CELL_SIZE//2, ball_y*self.CELL_SIZE + self.CELL_SIZE//2), self.CELL_SIZE//4)
        else:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(self.ball_pos_x), int(self.ball_pos_y)), self.CELL_SIZE//4)

        pygame.draw.rect(self.screen, (150, 150, 150), (0, self.net_top_position, self.CELL_SIZE, self.net_height))
        pygame.draw.rect(self.screen, (150, 150, 150), (self.SCREEN_WIDTH - self.CELL_SIZE, self.net_top_position, self.CELL_SIZE, self.net_height))

        pygame.display.flip()

    def run(self, buffer, get_actions_fn=None):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if get_actions_fn:
                actions = get_actions_fn(self)
            else:
                actions = self.random_actions()

            self.record_step(actions)
            self.step(actions)
            self.render()
            pygame.time.wait(50)
            if self.check_goal():
                self.reset_game()
            self.record_to_buffer(buffer)

        pygame.quit()

    def random_actions(self):
        action_choices = ['MOVE_LEFT', 'MOVE_RIGHT', 'MOVE_UP', 'MOVE_DOWN', 'PICK', 'SHOOT_UP', 'SHOOT_DOWN', 'SHOOT_LEFT', 'SHOOT_RIGHT']
        actions = {
            'A1': random.choice(action_choices),
            'A2': random.choice(action_choices),
            'B1': random.choice(action_choices),
            'B2': random.choice(action_choices),
        }
        return actions

    def action_to_index(action):
        action_mapping = {
            'MOVE_LEFT': 0, 
            'MOVE_RIGHT': 1, 
            'MOVE_UP': 2, 
            'MOVE_DOWN': 3, 
            'PICK': 4, 
            'SHOOT_LEFT': 5, 
            'SHOOT_RIGHT': 6, 
            'SHOOT_UP': 7, 
            'SHOOT_DOWN': 8
        }
        return action_mapping[action]


"""
if __name__ == "__main__":
    env = SoccerEnv()
    replay_buffer = ReplayBuffer(10000)
    env.run(replay_buffer)
"""
