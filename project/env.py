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
                 reward_alpha=0.0, scale_reward_by_entry_z=False,
                 tc_in_capital=False, exit_bonus=0.0, opportunity_cost=0.0):
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
        self._scale_reward_by_entry_z = scale_reward_by_entry_z
        self._tc_in_capital = tc_in_capital
        self._exit_bonus = exit_bonus
        self._opportunity_cost = opportunity_cost
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
            "macd":           spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "macd_signal":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "macd_hist":      spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "rolling_mean":   spaces.Box(low=0.0,     high=np.inf, shape=(1,), dtype=np.float32),
            "rolling_std":    spaces.Box(low=0.0,     high=np.inf, shape=(1,), dtype=np.float32),
        })
        
        self.action_space = spaces.Discrete(2) # 0 -> Maintain ; 1 -> Swap your position (either enter or exit)

        self._t = past_window # t is an index into the prices array. We start at window since we can't compute beta before that
    
    def _get_obs(self):
        macd, macd_signal, macd_hist = self._get_macd()
        rolling_mean, rolling_std = self._get_rolling_stats()
        return {
            "spread":         self._get_spread(),
            "spread_z_score": self._get_spread_z_score(),
            "entry_spread":   np.array([self._entry_spread],  dtype=np.float32),
            "entry_z_score":  np.array([self._entry_z_score], dtype=np.float32),
            "position":       self._in_position,
            "macd":           np.array([macd],         dtype=np.float32),
            "macd_signal":    np.array([macd_signal],  dtype=np.float32),
            "macd_hist":      np.array([macd_hist],    dtype=np.float32),
            "rolling_mean":   np.array([rolling_mean], dtype=np.float32),
            "rolling_std":    np.array([rolling_std],  dtype=np.float32),
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

    def _ema_series(self, values, span):
        k = 2.0 / (span + 1)
        ema = np.empty(len(values))
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = values[i] * k + ema[i - 1] * (1 - k)
        return ema

    def _get_macd(self):
        beta = self._get_beta()
        t0, t1 = self._t - self._past_window, self._t + 1
        s0 = np.log(self._pair_data.s[0][t0:t1])
        s1 = np.log(self._pair_data.s[1][t0:t1])
        abs_spread = np.abs(s0 - beta * s1 - self._spread_intercept)
        ema12 = self._ema_series(abs_spread, 12)
        ema26 = self._ema_series(abs_spread, 26)
        macd_series = ema12 - ema26
        signal_series = self._ema_series(macd_series, 9)
        macd = float(macd_series[-1])
        signal = float(signal_series[-1])
        return macd, signal, macd - signal

    def _get_rolling_stats(self):
        beta = self._get_beta()
        t0 = self._t - 50
        s0 = np.log(self._pair_data.s[0][t0:self._t])
        s1 = np.log(self._pair_data.s[1][t0:self._t])
        spread = s0 - beta * s1 - self._spread_intercept
        return abs(float(spread.mean())), float(spread.std())

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
        entry_capital = None
        entry_z_score = 0.0

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
                if self._tc_in_capital:
                    self._capital -= self._transaction_cost * self._last_trade_notional
                self._entry_capital = self._capital
                self._entry_z_score = float(self._get_spread_z_score()[0])
                self._entry_spread  = float(self._get_spread()[0])
            else:  # exiting position
                entry_capital = self._entry_capital
                entry_z_score = abs(self._entry_z_score)
                p0_now = self._pair_data.s[0][self._t]
                p1_now = self._pair_data.s[1][self._t]
                self._last_trade_notional = abs(self._position[0]) * p0_now + abs(self._position[1]) * p1_now
                self._capital = self._get_pnl()
                if self._tc_in_capital:
                    self._capital -= self._transaction_cost * self._last_trade_notional
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
                if self._scale_reward_by_entry_z:
                    reward *= entry_z_score
            elif self._in_position == 1:
                step_pnl = new_portfolio_value - old_portfolio_value
                reward = self._reward_alpha * max(step_pnl, 0.0)
            else:
                reward = 0.0
        elif self._sparse_reward:
            exited = (action == Actions.Swap.value and self._in_position == 0)
            if exited:
                trade_pnl = float(self._capital - entry_capital)
                reward = trade_pnl * abs(trade_pnl) if self._squared_reward else trade_pnl
                if self._scale_reward_by_entry_z:
                    reward *= entry_z_score
            else:
                reward = 0.0
        else:
            reward = new_portfolio_value - old_portfolio_value
            if self._exit_bonus > 0:
                exited = (action == Actions.Swap.value and self._in_position == 0)
                if exited:
                    trade_pnl = float(self._capital - entry_capital)
                    tc_cost = self._transaction_cost * self._last_trade_notional
                    reward += self._exit_bonus * max(0.0, trade_pnl - 4 * tc_cost)
            if self._opportunity_cost > 0:
                if self._in_position == 1:
                    current_z = abs(float(self._get_spread_z_score()[0]))
                    reward -= self._opportunity_cost * max(0.0, current_z - abs(self._entry_z_score))
                else:
                    current_z = abs(float(self._get_spread_z_score()[0]))
                    reward -= self._opportunity_cost * current_z

        if action == Actions.Swap.value and not self._tc_in_capital:
            reward -= self._transaction_cost * self._last_trade_notional

        return self._get_obs(), reward, terminated, False, {}