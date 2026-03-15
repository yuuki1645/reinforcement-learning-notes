"""
DQN エージェント（Experience Replay なし版）。

得た遷移 (s, a, r, s', done) をその場で 1 回だけ使い、Q ネットワークを更新する。
Replay バッファは使わず、オンラインで 1 ステップずつ学習する。終端（done）では次状態の Q を使わない。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 割引率: 将来の報酬をどれだけ重視するか (0〜1)
GAMMA = 0.99


def _to_tensor(x, dtype=torch.float32):
  """numpy 配列などを PyTorch テンソルに変換（dtype を統一）"""
  return torch.tensor(x, dtype=dtype)


class Agent:
  """Experience Replay を使わない DQN エージェント（Q ネットワーク 1 本）"""

  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions

    # Q ネットワーク: 状態 → 各行動の Q 値（CartPole なら 4 → 32 → 32 → 2）
    self.main_q_network = nn.Sequential(
      nn.Linear(num_states, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, num_actions),
    )

    self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.001)

  def get_action(self, state, episode):
    """ε-greedy で行動を選択する。エピソードが進むほど探索を減らす。"""
    epsilon = 0.5 * (1 / (episode + 1))
    if np.random.uniform(0, 1) < epsilon:
      return np.random.randint(0, self.num_actions)
    self.main_q_network.eval()
    with torch.no_grad():
      x = _to_tensor(state)
      return self.main_q_network(x).argmax().item()

  def update_main_q_network(self, state, action, reward, next_state, done):
    """
    遷移 1 件で Q ネットワークを 1 回だけ更新する（Experience Replay なし）。
    TD 目標: r + γ * (1 - done) * max_a' Q(s', a')。終端では次状態の Q を使わない。
    """
    self.main_q_network.train()

    x = _to_tensor(state)
    next_x = _to_tensor(next_state)

    # 現在の Q 値: Q(s, a)
    q_sa = self.main_q_network(x)[action]

    # 次状態の最大 Q 値（勾配を流さない）。終端なら TD 目標に含めない
    with torch.no_grad():
      next_q_max = self.main_q_network(next_x).max().item()
    td_target = reward + GAMMA * next_q_max * (1.0 - float(done))

    loss = F.smooth_l1_loss(q_sa, _to_tensor(td_target))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
