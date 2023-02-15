import pygame
import gymnasium as gym
from gymnasium import spaces
from random import randint
import numpy as np

class Grid:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.apple = None

    def spawn_apple(self, snake):
        x, y = randint(0, self.width-1), randint(0, self.height-1)

        while [x, y] in snake.tiles:            
            x, y = randint(0, self.width-1), randint(0, self.height-1)
        self.apple = [x, y]

class Snake:
    movements = {0: [1, 0], 1: [-1, 0], 2: [0, 1], 3: [0, -1]}
    
    def __init__(self, x, y, grid, random_spawn=True):
        self.size = 1
        self.x = x
        self.y = y

        
        if random_spawn:
            self.x = randint(0, grid.width-1)
            self.y = randint(0, grid.height-1)
        
        self.grid = grid
        self.tiles = [[self.x, self.y]]
        self.move_vec = Snake.movements[0]
        self.game_over = False
        self.score = 0
        self.block_next_frame = False
        self.grid.spawn_apple(self)
        
        self.steps_since_last_apple = 0
        self.steps_to_apple = np.abs(self.x-self.grid.apple[0]) + np.abs(self.y-self.grid.apple[1])
        
    def change_direction(self, direction):
        if not self.block_next_frame:
            move_vec = Snake.movements[direction]
            if not (self.move_vec == [-i for i in move_vec]):
                self.move_vec = move_vec
            self.block_next_frame = True

    def is_game_over(self):
        if self.x < 0 or self.x >= self.grid.width or self.y < 0 or self.y >= self.grid.height:
            return True
        if [self.x, self.y] in self.tiles:
            return True
        if len(self.tiles) == self.grid.width*self.grid.height:
            return True
        return False

    def consume_apple(self):
        self.tiles.append(self.tiles[-1])
        self.score += 1
        self.grid.apple = None

    def move(self):
        apple_collected = False
        self.x += self.move_vec[0]
        self.y += self.move_vec[1]  
        self.game_over = self.is_game_over()

        if self.game_over:
            return apple_collected

        if [self.x, self.y] == self.grid.apple:
            self.consume_apple() 
            apple_collected = True

        new_tiles = [[self.x, self.y]]

        for i in range(len(self.tiles)-1):
            new_tiles.append(self.tiles[i])

        self.tiles = new_tiles

        if not self.grid.apple:
            self.grid.spawn_apple(self)

        self.block_next_frame = False
        return apple_collected
            

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, width=5, height=5, window_width = 500, window_height = 500, fps=10):
        self.width = width
        self.height = height
        self.window_width = window_width
        self.window_height = window_height
        self.score = 0
        self.total_steps = 0
        self.metadata["render_fps"] = fps
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, max(width, height)-1, shape=(2,), dtype=int),
                #"grid": spaces.Box(0, 2, shape=(width*height,), dtype=int),
                "target": spaces.Box(0, max(width, height)-1, shape=(2,), dtype=int)
            }
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down", "nothing"
        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

    def _get_obs(self):
        grid = np.zeros((self.width, self.height), dtype=np.int32)
        for tile in self.snake.tiles[1:]:
            grid[tile] = 1
        grid[self.snake.tiles[0]] = 0
        grid[self._target_location] = 2
        
        return {"agent": self._agent_location, "target": self._target_location}#, "grid": grid.flatten()}

    def _get_info(self):
        return {
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.total_steps = 0
        self.snake = Snake(self.width//2, self.height//2, Grid(self.width, self.height))

        self._agent_location = np.array([self.snake.x, self.snake.y])
        self._target_location = np.array(self.snake.grid.apple)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in

        if action < 4:
            self.snake.change_direction(action)
        apple_collected = self.snake.move()
        self.total_steps += 1
        
        # An episode is done iff the agent has reached the target
        terminated = self.snake.game_over
        
        reward = int(apple_collected)
            
        self._agent_location = np.array([self.snake.x, self.snake.y])
        self._target_location = np.array(self.snake.grid.apple)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        for tile in self.snake.tiles:
            tile_x, tile_y = tile
            pygame.draw.rect(canvas, (100, 100, 100), [self.window_width/self.width*tile_x, self.window_height/self.height*tile_y, self.window_width/self.width, self.window_height/self.height], 0)
        pygame.draw.rect(canvas, (0, 0, 0), [self.window_width/self.width*self.snake.x, self.window_height/self.height*self.snake.y, self.window_width/self.width, self.window_height/self.height], 0)
        

        apple_x, apple_y = self.snake.grid.apple
        pygame.draw.ellipse(canvas, (255, 0, 0), [self.window_width/self.width*apple_x, self.window_height/self.height*apple_y, self.window_width/self.width, self.window_height/self.height], 2)
            

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
                
