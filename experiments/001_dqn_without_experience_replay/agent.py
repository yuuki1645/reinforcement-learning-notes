"""
DQN エージェント（Experience Replay なし版）

各遷移 (s, a, r, s') をその場で1回だけ使ってQネットワークを更新する。
バッファに貯めてミニバッチ学習する通常のDQNとは異なり、オンラインで1ステップずつ学習する。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 割引率: 将来の報酬をどれだけ重視するか (0〜1)
GAMMA = 0.99


def _to_tensor(x, dtype=torch.float32):
  """numpy配列をPyTorchテンソルに変換（dtypeを統一して学習を安定させる）"""
  return torch.tensor(x, dtype=dtype)


class Agent:
  """Experience Replay を使わない DQN エージェント"""

  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions

    # Qネットワーク: 状態 → 各行動のQ値
    # CartPole なら 状態4次元 → 隠れ32 → 隠れ32 → 行動2次元(Q値)
    self.model = nn.Sequential()
    self.model.add_module("fc1", nn.Linear(num_states, 32))
    self.model.add_module("relu1", nn.ReLU())
    self.model.add_module("fc2", nn.Linear(32, 32))
    self.model.add_module("relu2", nn.ReLU())
    self.model.add_module("fc3", nn.Linear(32, num_actions))

    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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

  def update(self, state, action, reward, next_state):
    """
    遷移 (state, action, reward, next_state) でQネットワークを1回だけ更新する。
    TD目標は r + γ * max_a Q(s', a) 。.item() で勾配を切っているので目標側は固定。
    """
    self.model.train()

    x = _to_tensor(state)
    next_x = _to_tensor(next_state)

    # 現在のQ値: Q(s, a)
    state_action_value = self.model(x)[action]

    # TD目標: r + γ * max_a' Q(s', a')  （s' のQは勾配を流さない）
    with torch.no_grad():
      next_q_max = self.model(next_x).max().item()
    expected_state_action_value = _to_tensor(reward + GAMMA * next_q_max)

    # Huber loss（外れ値に強い）で TD 誤差を最小化
    loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
