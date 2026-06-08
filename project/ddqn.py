import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s):
        return self.net(s)


class ReplayBuffer:
    """All tensors live on-device; sample() is a pure GPU/CPU gather — no copies."""

    def __init__(self, state_dim, max_size, device):
        self.max_size = max_size
        self.device   = device
        self.ptr  = 0
        self.size = 0

        self.s      = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.a      = torch.zeros((max_size, 1),         dtype=torch.long,    device=device)
        self.r      = torch.zeros((max_size, 1),         dtype=torch.float32, device=device)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.dw     = torch.zeros((max_size, 1),         dtype=torch.bool,    device=device)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr]      = torch.from_numpy(s).to(self.device)
        self.a[self.ptr]      = a
        self.r[self.ptr]      = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
        self.dw[self.ptr]     = dw

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

    def __len__(self):
        return self.size


class DoubleDQN:
    """
    Double DQN with soft target updates, Huber loss, gradient clipping,
    and linear epsilon annealing.

    Target: r + γ * Q_target(s', argmax_a Q_online(s', a))
    """

    def __init__(
        self,
        state_dim:       int,
        action_dim:      int   = 3,
        hidden:          int   = 64,
        lr:              float = 3e-4,
        gamma:           float = 0.99,
        tau:             float = 0.005,
        buffer_size:     int   = int(1e6),
        batch_size:      int   = 256,
        eps_start:       float = 1.0,
        eps_end:         float = 0.05,
        eps_decay_steps: int   = 200_000,
        grad_clip:       float = 1.0,
        device:          str   = "cpu",
        # backward-compat alias
        exp_noise:       float = None,
    ):
        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.grad_clip  = grad_clip
        self.device     = torch.device(device)

        if exp_noise is not None:
            # legacy: fixed epsilon
            self.eps_start       = exp_noise
            self.eps_end         = exp_noise
            self.eps_decay_steps = 1
        else:
            self.eps_start       = eps_start
            self.eps_end         = eps_end
            self.eps_decay_steps = max(1, eps_decay_steps)

        self._act_steps = 0

        self.q_net    = QNet(state_dim, action_dim, hidden).to(self.device)
        self.q_target = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.optimizer     = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(state_dim, buffer_size, self.device)

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._act_steps / self.eps_decay_steps)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic:
            self._act_steps += 1
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            if deterministic or np.random.rand() >= self.epsilon:
                return int(self.q_net(s).argmax().item())
            return np.random.randint(self.action_dim)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Double DQN: online picks action, target evaluates it
            argmax_a   = self.q_net(s_next).argmax(dim=1, keepdim=True)
            max_q_next = self.q_target(s_next).gather(1, argmax_a)
            target_Q   = r + (~dw) * self.gamma * max_q_next

        current_Q = self.q_net(s).gather(1, a)
        loss = F.smooth_l1_loss(current_Q, target_Q)   # Huber

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft target update
        for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return loss.item()

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False
