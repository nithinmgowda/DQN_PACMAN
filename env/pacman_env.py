import gymnasium as gym
import numpy as np
import pygame
import sys
from gymnasium import spaces
from typing import Optional, Tuple, Any
import random
from config import *

class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        
        # Observation space is a 2D grid representing the game state
        # Channels: Pacman, Ghosts, Food, Power pellets, Walls
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(5, SCREEN_HEIGHT // CELL_SIZE, SCREEN_WIDTH // CELL_SIZE),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Game state
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Initialize game state
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.power_pellet_timer = 0
        
        # Initialize Pacman position
        self.pacman_pos = np.array([SCREEN_HEIGHT // (2 * CELL_SIZE),
                                  SCREEN_WIDTH // (2 * CELL_SIZE)])
        
        # Initialize ghosts
        self.ghosts = []
        for _ in range(GHOST_COUNT):
            self.ghosts.append(self._spawn_ghost())
        
        # Initialize food and power pellets
        self.food = self._initialize_food()
        self.power_pellets = self._initialize_power_pellets()
        
        # Get initial observation
        observation = self._get_observation()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Execute action
        self.steps += 1
        reward = STEP_PENALTY
        
        # Move Pacman
        new_pos = self.pacman_pos.copy()
        if action == 0:  # UP
            new_pos[0] -= 1
        elif action == 1:  # DOWN
            new_pos[0] += 1
        elif action == 2:  # LEFT
            new_pos[1] -= 1
        elif action == 3:  # RIGHT
            new_pos[1] += 1
        
        # Check if move is valid
        if self._is_valid_position(new_pos):
            self.pacman_pos = new_pos
        
        # Check for food collection
        food_pos = tuple(self.pacman_pos)
        if food_pos in self.food:
            self.food.remove(food_pos)
            reward += PELLET_REWARD
            self.score += PELLET_REWARD
        
        # Check for power pellet collection
        if food_pos in self.power_pellets:
            self.power_pellets.remove(food_pos)
            self.power_pellet_timer = POWER_PELLET_DURATION
            reward += POWER_PELLET_REWARD
            self.score += POWER_PELLET_REWARD
        
        # Update power pellet timer
        if self.power_pellet_timer > 0:
            self.power_pellet_timer -= 1
        
        # Move ghosts and check for collisions
        self._update_ghosts()
        
        # Check for game over conditions
        if self._check_collision():
            if self.power_pellet_timer > 0:
                reward += GHOST_REWARD
                self.score += GHOST_REWARD
                # Respawn eaten ghost
                self.ghosts = [g for g in self.ghosts if not np.array_equal(g, self.pacman_pos)]
                self.ghosts.append(self._spawn_ghost())
            else:
                reward += DEATH_PENALTY
                self.game_over = True
        
        # Check win condition
        terminated = self.game_over or len(self.food) == 0
        truncated = self.steps >= 1000  # Maximum episode length
        
        observation = self._get_observation()
        info = {"score": self.score}
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((5, SCREEN_HEIGHT // CELL_SIZE, SCREEN_WIDTH // CELL_SIZE), dtype=np.float32)
        
        # Pacman layer
        obs[0, self.pacman_pos[0], self.pacman_pos[1]] = 1
        
        # Ghost layer
        for ghost in self.ghosts:
            obs[1, ghost[0], ghost[1]] = 1
        
        # Food layer
        for food in self.food:
            obs[2, food[0], food[1]] = 1
        
        # Power pellet layer
        for pellet in self.power_pellets:
            obs[3, pellet[0], pellet[1]] = 1
        
        # Wall layer (to be implemented)
        # obs[4] = self.walls
        
        return obs

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        return (0 <= pos[0] < SCREEN_HEIGHT // CELL_SIZE and
                0 <= pos[1] < SCREEN_WIDTH // CELL_SIZE)

    def _spawn_ghost(self) -> np.ndarray:
        while True:
            pos = np.array([
                random.randint(0, SCREEN_HEIGHT // CELL_SIZE - 1),
                random.randint(0, SCREEN_WIDTH // CELL_SIZE - 1)
            ])
            if not np.array_equal(pos, self.pacman_pos):
                return pos

    def _initialize_food(self) -> set:
        food = set()
        for i in range(1, (SCREEN_HEIGHT // CELL_SIZE) - 1):
            for j in range(1, (SCREEN_WIDTH // CELL_SIZE) - 1):
                if random.random() < 0.3:  # 30% chance of food
                    food.add((i, j))
        return food

    def _initialize_power_pellets(self) -> set:
        pellets = set()
        corners = [
            (1, 1),
            (1, (SCREEN_WIDTH // CELL_SIZE) - 2),
            ((SCREEN_HEIGHT // CELL_SIZE) - 2, 1),
            ((SCREEN_HEIGHT // CELL_SIZE) - 2, (SCREEN_WIDTH // CELL_SIZE) - 2)
        ]
        for corner in corners:
            pellets.add(corner)
        return pellets

    def _update_ghosts(self):
        for i, ghost in enumerate(self.ghosts):
            # Simple ghost movement: random direction
            direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            new_pos = ghost + direction
            if self._is_valid_position(new_pos):
                self.ghosts[i] = new_pos

    def _check_collision(self) -> bool:
        return any(np.array_equal(self.pacman_pos, ghost) for ghost in self.ghosts)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Pac-Man RL")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill(BLACK)

        # Draw food
        for food in self.food:
            pygame.draw.circle(canvas, WHITE,
                             (food[1] * CELL_SIZE + CELL_SIZE // 2,
                              food[0] * CELL_SIZE + CELL_SIZE // 2),
                             CELL_SIZE // 6)

        # Draw power pellets
        for pellet in self.power_pellets:
            pygame.draw.circle(canvas, WHITE,
                             (pellet[1] * CELL_SIZE + CELL_SIZE // 2,
                              pellet[0] * CELL_SIZE + CELL_SIZE // 2),
                             CELL_SIZE // 3)

        # Draw Pac-Man
        pygame.draw.circle(canvas, YELLOW,
                         (self.pacman_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                          self.pacman_pos[0] * CELL_SIZE + CELL_SIZE // 2),
                         CELL_SIZE // 2)

        # Draw ghosts
        ghost_color = BLUE if self.power_pellet_timer == 0 else RED
        for ghost in self.ghosts:
            pygame.draw.circle(canvas, ghost_color,
                             (ghost[1] * CELL_SIZE + CELL_SIZE // 2,
                              ghost[0] * CELL_SIZE + CELL_SIZE // 2),
                             CELL_SIZE // 2)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
