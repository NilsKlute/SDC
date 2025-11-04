import gymnasium as gym
import numpy as np
from gymnasium.core import ActType

class SDC_Wrapper(gym.Wrapper):
    def __init__(self, env, remove_score=True, return_linear_velocity=False):
        super().__init__(env)

        self.return_linear_velocity = return_linear_velocity
        self.remove_score = remove_score

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)

        if self.remove_score:
            observation[84:, :11, :] = 0

        if self.return_linear_velocity:
            info['speed'] = np.linalg.norm(self.car.hull.linearVelocity)

        return observation, info

    def step(self, action: ActType):
        observation, reward, done, truncated, info = super().step(action)
        reward_clipped = np.clip(reward, -0.1, 1e8)

        if self.remove_score:
            observation[84:, :11, :] = 0

        if self.return_linear_velocity:
            info['speed'] = np.linalg.norm(self.car.hull.linearVelocity)

        return observation, reward_clipped, done, truncated, info