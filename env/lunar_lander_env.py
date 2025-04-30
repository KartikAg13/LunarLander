import gym
import numpy as np
from typing import Tuple

# Create and return a LunarLander-v2 environment
def make_env(render_mode='human'):
	env = gym.make('LunarLander-v2', render_mode=render_mode)

	# Compatibility patch for numpy.bool8 if needed
	if not hasattr(np, 'bool8'):
		np.bool8 = np.bool_

	return env

# Reset the environment and return initial observation
def reset_env(env) -> np.ndarray:
	state, _ = env.reset()
	return state

# Take an action in the environment
def step_env(env, action) -> Tuple[np.ndarray, float, bool]:
	next_state, reward, terminated, truncated, info = env.step(action)
	done = bool(terminated or truncated)

	return (next_state, reward, done)
