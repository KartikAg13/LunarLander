# dqn_lunar_lander_env_setup.py
import gym
import numpy as np

# Compatibility patch: define np.bool8 if missing for Gym
if not hasattr(np, 'bool8'):
	np.bool8 = np.bool_

# 1. Initialize the Lunar Lander environment with human rendering via pyglet
env = gym.make('LunarLander-v2', render_mode='human')

# 2. Inspect state & action spaces
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

# 3. Run a random-action episode to verify setup
def run_random_episode():
	state, info = env.reset()           # return obs, info in gym v0.26+
	done = False
	total_reward = 0.0

	while not done:
		action = env.action_space.sample()  # random policy
		next_state, reward, terminated, truncated, info = env.step(action)
		done = bool(terminated or truncated)
		total_reward += reward

	print(f"Episode finished. Total reward: {total_reward:.2f}")
	input("Press Enter to close the window...")
	env.close()