import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dueling network ───────────────────────────────────────────────────────────

class DuelingQNet(nn.Module):
    """
    Dueling Network (Wang et al. 2016).

    shared → V(s) stream  +  A(s,a) stream
    Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

    Separates how good the state is from which action is best — helps in
    states where action choice barely matters (e.g. spread near zero).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden, 1)
        self.adv_stream   = nn.Linear(hidden, action_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = self.shared(s)
        V = self.value_stream(h)                       # (B, 1)
        A = self.adv_stream(h)                         # (B, action_dim)
        return V + A - A.mean(dim=1, keepdim=True)     # (B, action_dim)


# ── Sum-tree for PER ──────────────────────────────────────────────────────────

class _SumTree:
    """
    Binary tree where each leaf stores a priority and each internal node
    stores the sum of its children.  O(log n) update and stratified sample.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree    = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._ptr     = 0
        self.size     = 0

    def _propagate(self, idx: int, delta: float):
        while idx > 0:
            idx = (idx - 1) >> 1
            self._tree[idx] += delta

    def update(self, leaf_idx: int, priority: float):
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def add(self, priority: float) -> int:
        leaf_idx = self._ptr + self.capacity - 1
        self.update(leaf_idx, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return leaf_idx

    def get(self, value: float) -> int:
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self._tree):
                return idx
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1

    @property
    def total(self) -> float:
        return float(self._tree[0])


# ── Prioritized replay buffer ─────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Proportional PER (Schaul et al. 2016).

    Sampling probability:  P(i) = p_i^α / Σ p_j^α
    IS correction weight:  w_i  = (N · P(i))^{−β}  (normalised by max w)
    β anneals linearly from beta_start → 1.0 over beta_steps train calls.
    New transitions are inserted with the current maximum priority so they
    are guaranteed to be replayed at least once.
    """

    def __init__(
        self,
        state_dim:  int,
        max_size:   int,
        device:     torch.device,
        alpha:      float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int   = 200_000,
    ):
        self.max_size   = max_size
        self.device     = device
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_steps = max(1, beta_steps)
        self._beta_step = 0
        self._eps       = 1e-6
        self._max_prio  = 1.0

        self._tree = _SumTree(max_size)

        self.s      = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.a      = torch.zeros((max_size, 1),         dtype=torch.long,    device=device)
        self.r      = torch.zeros((max_size, 1),         dtype=torch.float32, device=device)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.dw     = torch.zeros((max_size, 1),         dtype=torch.bool,    device=device)

    def add(self, s, a, r, s_next, dw):
        leaf_idx = self._tree.add(self._max_prio ** self.alpha)
        data_idx = leaf_idx - (self._tree.capacity - 1)

        self.s[data_idx]      = torch.from_numpy(s).to(self.device)
        self.a[data_idx]      = a
        self.r[data_idx]      = r
        self.s_next[data_idx] = torch.from_numpy(s_next).to(self.device)
        self.dw[data_idx]     = dw

    def sample(self, batch_size: int):
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self._beta_step / self.beta_steps)
        self._beta_step += 1

        segment = self._tree.total / batch_size
        leaf_indices = np.empty(batch_size, dtype=np.int64)
        data_indices = np.empty(batch_size, dtype=np.int64)
        priorities   = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            v = np.random.uniform(segment * i, segment * (i + 1))
            leaf_idx = self._tree.get(v)
            leaf_indices[i] = leaf_idx
            data_indices[i] = leaf_idx - (self._tree.capacity - 1)
            priorities[i]   = self._tree._tree[leaf_idx]

        probs   = priorities / self._tree.total
        weights = (self._tree.size * probs) ** (-beta)
        weights /= weights.max()
        weights  = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        idx = torch.tensor(data_indices, device=self.device)
        return (self.s[idx], self.a[idx], self.r[idx],
                self.s_next[idx], self.dw[idx],
                leaf_indices, weights)

    def update_priorities(self, leaf_indices: np.ndarray, td_errors: np.ndarray):
        for leaf_idx, td_err in zip(leaf_indices, td_errors):
            prio = (abs(float(td_err)) + self._eps) ** self.alpha
            self._tree.update(int(leaf_idx), prio)
            if prio > self._max_prio:
                self._max_prio = prio

    def __len__(self):
        return self._tree.size


# ── Double DQN agent ──────────────────────────────────────────────────────────

class DoubleDQN:
    """
    Double DQN + Dueling Network + Prioritized Experience Replay.
    Soft target updates (Polyak τ), Huber loss, grad clipping,
    linear epsilon annealing.

    Drop-in replacement for ddqn.py — same constructor signature
    plus optional PER hyperparams.
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
        # PER
        per_alpha:       float = 0.6,
        per_beta_start:  float = 0.4,
        per_beta_steps:  int   = None,   # defaults to eps_decay_steps
        device:          str   = "cpu",
        exp_noise:       float = None,   # backward-compat alias
    ):
        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.grad_clip  = grad_clip
        self.device     = torch.device(device)

        if exp_noise is not None:
            self.eps_start       = exp_noise
            self.eps_end         = exp_noise
            self.eps_decay_steps = 1
        else:
            self.eps_start       = eps_start
            self.eps_end         = eps_end
            self.eps_decay_steps = max(1, eps_decay_steps)

        beta_steps = per_beta_steps if per_beta_steps is not None else self.eps_decay_steps
        self._act_steps = 0

        self.q_net    = DuelingQNet(state_dim, action_dim, hidden).to(self.device)
        self.q_target = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.optimizer     = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(
            state_dim, buffer_size, self.device,
            alpha=per_alpha, beta_start=per_beta_start, beta_steps=beta_steps,
        )

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

        s, a, r, s_next, dw, leaf_indices, weights = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            argmax_a   = self.q_net(s_next).argmax(dim=1, keepdim=True)
            max_q_next = self.q_target(s_next).gather(1, argmax_a)
            target_Q   = r + (~dw) * self.gamma * max_q_next

        current_Q = self.q_net(s).gather(1, a)

        # IS-weighted Huber loss
        elementwise_loss = F.smooth_l1_loss(current_Q, target_Q, reduction='none')
        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Update priorities with new TD errors
        td_errors = (target_Q - current_Q).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(leaf_indices, td_errors)

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
