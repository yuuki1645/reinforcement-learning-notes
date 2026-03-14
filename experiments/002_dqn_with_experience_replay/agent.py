"""
DQN エージェント（Experience Replay あり版）

Replay バッファに貯めた遷移をランダムにサンプリングし、
ミニバッチで TD 学習する。終端状態（done）では次状態の Q を使わない。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_memory import ReplayMemory


# 1回の更新で使う遷移数
BATCH_SIZE = 32
# 割引率: 将来の報酬をどれだけ重視するか (0〜1)
GAMMA = 0.99


def _to_tensor(x, dtype=torch.float32):
  """numpy 配列などを PyTorch テンソルに変換（dtype を統一）"""
  return torch.tensor(x, dtype=dtype)


class Agent:
  """Experience Replay を使う DQN エージェント"""

  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions

    # Q ネットワーク: 状態 → 各行動の Q 値（CartPole なら 4 → 32 → 32 → 2）
    self.model = nn.Sequential()
    self.model.add_module("fc1", nn.Linear(num_states, 32))
    self.model.add_module("relu1", nn.ReLU())
    self.model.add_module("fc2", nn.Linear(32, 32))
    self.model.add_module("relu2", nn.ReLU())
    self.model.add_module("fc3", nn.Linear(32, num_actions))

    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    self.memory = ReplayMemory(capacity=10000)

  def get_action(self, state, episode):
    """
    ε-greedy で行動を選択する。
    エピソードが進むほど ε を小さくし、探索から活用へ移行する。
    """
    epsilon = 0.5 * (1 / (episode + 1))

    if np.random.uniform(0, 1) < epsilon:
      return np.random.randint(0, self.num_actions)
    else:
      self.model.eval()
      with torch.no_grad():
        x = _to_tensor(state)
        q = self.model(x)
        return q.argmax().item()

  def memorize(self, state, action, reward, next_state, done):
    """遷移 (s, a, r, s', done) を Replay バッファに追加する。"""
    self.memory.push(state, action, reward, next_state, done)

  def update(self):
    """
    Replay バッファから BATCH_SIZE 件サンプルし、TD 誤差で Q ネットワークを更新する。
    TD 目標: r + γ * (1 - done) * max_a' Q(s', a')  （done のときは r のみ）
    """
    if len(self.memory) < BATCH_SIZE:
      return

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)

    # 現在の Q 値: Q(s_t, a_t)。gather で選んだ行動の Q だけ取り出す
    state_action_values = self.model(state_batch).gather(dim=1, index=action_batch)

    # 次状態の最大 Q 値（勾配を流さない）。終端なら使わない
    with torch.no_grad():
      next_state_action_values = self.model(next_state_batch).max(dim=1)[0]
    # TD 目標: 終端でないときだけ γ * max Q(s') を足す
    expected_state_action_values = reward_batch + GAMMA * next_state_action_values * (1.0 - done_batch)

    self.model.train()
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()