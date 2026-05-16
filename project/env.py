import gymnasium as gym
from gymnasium import spaces
import numpy as np
from data_loader import PairData
from enum import Enum

class Actions(Enum):
    Maintain = 0
    Swap = 1
    
class Market(gym.Env):
    def __init__ (self, pair_data: PairData, past_window=60, initial_capital = 10000):
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._position = np.array([0,0]) #  actual current position in the real stocks
        
        self._pair_data = pair_data
        self._past_window = past_window # how many ticks in the past we look to compute spread
        self._in_position = 0 # 0 if currently outside the position, 1 if currently in the position
        
        self.observation_space = spaces.Dict(
            {
            "spread" : spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)    ,
            "position" : spaces.Discrete(2)  # 0 = flat, 1 = in position
        })
        
        self.action_space = spaces.Discrete(2) # 0 -> Maintain ; 1 -> Swap your position (either enter or exit)

        self._t = past_window # t is an index into the prices array. We start at window since we can't compute beta before that
    
    def _get_obs(self):
        return {
            "spread": self._get_spread(),
            "spread_z_score": self._get_spread_z_score(),
            "position": self._in_position
        }
        
    def _get_spread(self):
        # Run an OLS on the past "_past_window" ticks to compute a cointegration beta between the two stocks
        s0 = self._pair_data.s[0][self._t - self._past_window:self._t]
        s1 = self._pair_data.s[1][self._t - self._past_window:self._t]
        
        beta = np.polyfit(s1, s0, 1)[0]
        
        spread = self._pair_data.s[0][self._t] - beta * self._pair_data.s[1][self._t]
        
        return np.array([spread], dtype=np.float32)

    def _get_spread_z_score(self):
        s0 = self._pair_data.s[0][self._t - self._past_window:self._t]
        s1 = self._pair_data.s[1][self._t - self._past_window:self._t]
        
        beta = np.polyfit(s1, s0, 1)[0]

        spread_window = s0 - beta * s1
        current_spread = self._pair_data.s[0][self._t] - beta * self._pair_data.s[1][self._t]
        zscore = (current_spread - spread_window.mean()) / (spread_window.std() + 1e-8)
        return np.array([zscore], dtype=np.float32)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self._t = self._past_window
        self._in_position = 0
        self._position = np.zeros(2)
        self._capital = self._initial_capital
        
        return self._get_obs(), {}
    
    def _get_pnl(self):
        # returns current portfolio value
        return self._capital + np.dot(self._position, np.array([self._pair_data.s[0][self._t], self._pair_data.s[1][self._t]]))
        
    def step(self, action):
        old_portfolio_value = self._get_pnl()
        
        if action == Actions.Swap.value:
            if self._pair_data.s[0][self._t] > self._pair_data.s[1][self._t]:
                cheap_i = 1
                expensive_i = 0
            else:
                cheap_i = 0
                expensive_i = 1
                
            # dollar neutral approach -> Sepnd half of our money on the short leg, half on the long leg
            if self._in_position == 0:  # entering position
                half = self._capital / 2
                self._position[cheap_i] = half / self._pair_data.s[cheap_i][self._t]
                self._position[expensive_i] = -(half / self._pair_data.s[expensive_i][self._t])
            else:  # exiting position
                self._capital = self._get_pnl()
                self._position = np.zeros(2)
                
            self._in_position = 1 - self._in_position

        self._t += 1
        
        terminated = self._t >= len(self._pair_data) - 1
        new_portfolio_value = self._get_pnl()
        reward = new_portfolio_value - old_portfolio_value
    
    
        return self._get_obs(), reward, terminated, False, {}