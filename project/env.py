import gymnasium as gym
from gymnasium import spaces
import numpy as np
from data_loader import PairData
from enum import Enum

class Actions(Enum):
    Maintain = 0
    Swap = 1
    
class Market(gym.Env):
    def __init__(self, pair_data: PairData, past_window=60, initial_capital=10000,
                 fixed_beta=None, spread_intercept=0.0, fixed_std=None,
                 use_kalman=False, kf_Q=1e-5, kf_R=1e-3,
                 sparse_reward=False, transaction_cost=0.0, squared_reward=False,
                 reward_alpha=0.0):
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._position = np.array([0, 0])
        self._fixed_beta = fixed_beta
        self._spread_intercept = spread_intercept
        self._fixed_std = fixed_std
        self._use_kalman = use_kalman
        self._kf_Q = kf_Q
        self._kf_R = kf_R
        self._kf_beta = None
        self._kf_P = 1.0
        self._position_beta = None
        self._sparse_reward = sparse_reward
        self._transaction_cost = transaction_cost
        self._squared_reward = squared_reward
        self._reward_alpha = reward_alpha
        self._entry_capital = None
        self._entry_z_score = 0.0
        self._entry_spread  = 0.0
        self._last_trade_notional = 0.0

        self._pair_data = pair_data
        self._past_window = past_window
        self._in_position = 0

        self.observation_space = spaces.Dict({
            "spread":         spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "spread_z_score": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "entry_spread":   spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "entry_z_score":  spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "position":       spaces.Discrete(2),
        })
        
        self.action_space = spaces.Discrete(2) # 0 -> Maintain ; 1 -> Swap your position (either enter or exit)

        self._t = past_window # t is an index into the prices array. We start at window since we can't compute beta before that
    
    def _get_obs(self):
        return {
            "spread":         self._get_spread(),
            "spread_z_score": self._get_spread_z_score(),
            "entry_spread":   np.array([self._entry_spread],  dtype=np.float32),
            "entry_z_score":  np.array([self._entry_z_score], dtype=np.float32),
            "position":       self._in_position,
        }
        
    def _get_beta(self):
        if self._position_beta is not None:
            return self._position_beta
        if self._fixed_beta is not None:
            return self._fixed_beta
        if self._use_kalman:
            return self._kf_beta
        s0 = np.log(self._pair_data.s[0][self._t - self._past_window:self._t])
        s1 = np.log(self._pair_data.s[1][self._t - self._past_window:self._t])
        return np.polyfit(s1, s0, 1)[0]

    def _kalman_update(self):
        s0 = np.log(self._pair_data.s[0][self._t])
        s1 = np.log(self._pair_data.s[1][self._t])
        P_pred = self._kf_P + self._kf_Q
        K = P_pred * s1 / (s1 ** 2 * P_pred + self._kf_R)
        self._kf_beta = self._kf_beta + K * (s0 - s1 * self._kf_beta)
        self._kf_P = (1 - K * s1) * P_pred

    def _raw_spread(self, t):
        beta = self._get_beta()
        return np.log(self._pair_data.s[0][t]) - beta * np.log(self._pair_data.s[1][t]) - self._spread_intercept

    def _get_spread(self):
        return np.array([self._raw_spread(self._t)], dtype=np.float32)

    def _get_spread_z_score(self):
        if self._fixed_std is not None:
            zscore = self._raw_spread(self._t) / (self._fixed_std + 1e-8)
        else:
            beta = self._get_beta()
            s0 = np.log(self._pair_data.s[0][self._t - self._past_window:self._t])
            s1 = np.log(self._pair_data.s[1][self._t - self._past_window:self._t])
            spread_window = s0 - beta * s1 - self._spread_intercept
            zscore = self._raw_spread(self._t) / (spread_window.std() + 1e-8)
        return np.array([zscore], dtype=np.float32)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self._t = self._past_window
        self._in_position = 0
        self._position = np.zeros(2)
        self._capital = self._initial_capital
        self._position_beta = None
        self._entry_capital = None
        self._entry_z_score = 0.0
        self._entry_spread  = 0.0
        self._last_trade_notional = 0.0

        if self._use_kalman:
            s0 = np.log(self._pair_data.s[0][:self._past_window])
            s1 = np.log(self._pair_data.s[1][:self._past_window])
            self._kf_beta = np.polyfit(s1, s0, 1)[0]
            self._kf_P = 1.0

        return self._get_obs(), {}
    
    def _get_pnl(self):
        # returns current portfolio value
        return self._capital + np.dot(self._position, np.array([self._pair_data.s[0][self._t], self._pair_data.s[1][self._t]]))
        
    def step(self, action):
        old_portfolio_value = self._get_pnl()
        
        if action == Actions.Swap.value:
            if self._in_position == 0:  # entering position
                self._position_beta = self._get_beta()
                beta = self._position_beta
                p0   = self._pair_data.s[0][self._t]
                p1   = self._pair_data.s[1][self._t]
                C    = self._capital
                # beta-neutral sizing: D0 = C/(1+β), D1 = β·C/(1+β)
                d0 = C / (1 + beta)
                d1 = beta * C / (1 + beta)
                self._last_trade_notional = d0 + d1
                # spread > 0 → s0 overpriced: short s0, long s1
                if self._raw_spread(self._t) > 0:
                    self._position[0] = -(d0 / p0)
                    self._position[1] =  (d1 / p1)
                    self._capital += d0 - d1  # receive short proceeds, pay for long
                else:
                    self._position[0] =  (d0 / p0)
                    self._position[1] = -(d1 / p1)
                    self._capital += d1 - d0  # receive short proceeds, pay for long
                self._entry_capital = self._capital
                self._entry_z_score = float(self._get_spread_z_score()[0])
                self._entry_spread  = float(self._get_spread()[0])
            else:  # exiting position
                entry_capital = self._entry_capital
                p0_now = self._pair_data.s[0][self._t]
                p1_now = self._pair_data.s[1][self._t]
                self._last_trade_notional = abs(self._position[0]) * p0_now + abs(self._position[1]) * p1_now
                self._capital = self._get_pnl()
                self._position = np.zeros(2)
                self._position_beta = None
                self._entry_capital = None
                self._entry_z_score = 0.0
                self._entry_spread  = 0.0

            self._in_position = 1 - self._in_position

        self._t += 1

        if self._use_kalman:
            self._kalman_update()

        terminated = self._t >= len(self._pair_data) - 1
        new_portfolio_value = self._get_pnl()

        if self._reward_alpha > 0:
            exited = (action == Actions.Swap.value and self._in_position == 0)
            if exited:
                trade_pnl = float(self._capital - entry_capital)
                reward = trade_pnl * abs(trade_pnl) if self._squared_reward else trade_pnl
            elif self._in_position == 1:
                step_pnl = new_portfolio_value - old_portfolio_value
                reward = self._reward_alpha * step_pnl
            else:
                reward = 0.0
        elif self._sparse_reward:
            exited = (action == Actions.Swap.value and self._in_position == 0)
            if exited:
                trade_pnl = float(self._capital - entry_capital)
                reward = trade_pnl * abs(trade_pnl) if self._squared_reward else trade_pnl
            else:
                reward = 0.0
        else:
            reward = new_portfolio_value - old_portfolio_value

        if action == Actions.Swap.value:
            reward -= self._transaction_cost * self._last_trade_notional

        return self._get_obs(), reward, terminated, False, {}