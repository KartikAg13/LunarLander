from typing import override
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import pygame


class DQN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return F.softmax(self.fc3(x))


class GridEnv(gym.Env):
    def __init__(self, size: int, mode: str | None) -> None:
        super(GridEnv, self).__init__()
        # size of the grid
        self.grid_size = size
        self.render_mode = mode
        self.cell_size = 20
        self.window_size = self.grid_size * self.cell_size

        # spawn locations
        self.snake1_location = np.array([-1, -1], dtype=np.int32)
        self.snake2_location = np.array([-1, -1], dtype=np.int32)
        self.fruit1_location = np.array([-1, -1], dtype=np.int32)
        self.fruit2_location = np.array([-1, -1], dtype=np.int32)

        # observation space
        self.surrounding_space = gym.spaces.Box(-1, 1, (4, 0), np.int32)
        self.distance_space = gym.spaces.Box(0, 2 * size, (2, 0), np.int32)
        self.observation_space = gym.spaces.Dict(
            {
                "surroundings": self.surrounding_space,
                "distances": self.distance_space,
            }
        )

        # action space
        self.action_space = gym.spaces.Discrete(4)
        self.direction = {
            0: np.array([-1, 0]),  # left
            1: np.array([0, 1]),  # up
            2: np.array([1, 0]),  # right
            3: np.array([0, -1]),  # down
        }

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.window_size * 2 + 50, self.window_size + 100)
            )
            pygame.display.set_caption("Snake RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)

        self.reset()

    def _get_obs(self):
        # returns the current state
        return {
            "snake1": self.snake1_location,
            "snake2": self.snake2_location,
            "fruit1": self.fruit1_location,
            "fruit2": self.fruit2_location,
        }

    def _get_info(self):
        # returns the distance between snake and fruit
        return {
            "distance11": np.linalg.norm(
                self.snake1_location - self.fruit1_location, ord=1
            ),
            "distance12": np.linalg.norm(
                self.snake1_location - self.fruit2_location, ord=1
            ),
            "distance21": np.linalg.norm(
                self.snake2_location - self.fruit1_location, ord=1
            ),
            "distance22": np.linalg.norm(
                self.snake2_location - self.fruit2_location, ord=1
            ),
        }

    @override
    def reset(self, *, seed: int | None = None):
        # make sure the env is reproducible
        super().reset(seed=seed)

        # spawn the agent
        self.snake1_location = self.np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32
        )
        self.snake2_location = self.snake1_location
        while np.array_equal(self.snake1_location, self.snake2_location):
            self.snake2_location = self.np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32
            )

        self.snake1 = {
            "location": self.snake1_location,
            "score": 0,
            "color": (0, 255, 0),
        }
        self.snake2 = {
            "location": self.snake2_location,
            "score": 0,
            "color": (0, 0, 255),
        }

        # spawn the target
        self.fruit1_location = self.spawn_fruits(1)
        self.fruit2_location = self.spawn_fruits(2)
        self.fruit_color = (255, 0, 0)

        self.steps: int = 0
        self.max_steps: int = 500
        self.time_left: int = 60
        self.frame_count: int = 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action1: int, action2: int):
        # take the action and update location
        direction = self.direction[action1]
        self.snake1_location = np.clip(
            a=self.snake1_location + direction,
            a_min=0,
            a_max=self.grid_size - 1,
        )
        direction = self.direction[action2]
        self.snake2_location = np.clip(
            a=self.snake2_location + direction,
            a_min=0,
            a_max=self.grid_size - 1,
        )

    def spawn_fruits(self, spawn: int):
        while True:
            fruit = self.np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32
            )
            if np.array_equal(fruit, self.snake1_location) or np.array_equal(
                fruit, self.snake2_location
            ):
                continue
            else:
                if (
                    spawn == 1 and np.array_equal(fruit, self.fruit2_location)
                ) or (
                    spawn == 2 and np.array_equal(fruit, self.fruit1_location)
                ):
                    continue
                else:
                    break
        return fruit


def main():
    # gym.register("gymnasium_env/GridEnv-v0", GridEnv, 500)
    ...
