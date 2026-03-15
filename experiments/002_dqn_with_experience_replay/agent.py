"""
DQN エージェント（Experience Replay あり版）。

Replay からミニバッチをサンプルし、1 本の Q ネットワークだけで TD 学習する。
ターゲットネットワークは使わない。終端（done）では次状態の Q を使わない。
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
  """Experience Replay を使う DQN エージェント（Q ネットワーク 1 本）"""

  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions
    self.memory = ReplayMemory(capacity=10000)

    # Q ネットワーク: 状態 → 各行動の Q 値（CartPole なら 4 → 32 → 32 → 2）
    self.main_q_network = nn.Sequential(
      nn.Linear(num_states, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, num_actions),
    )

    self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

  def get_action(self, state, episode):
    """ε-greedy で行動を選択する。エピソードが進むほど探索を減らす。"""
    epsilon = 0.5 * (1 / (episode + 1))
    if np.random.uniform(0, 1) < epsilon:
      return np.random.randint(0, self.num_actions)
    self.main_q_network.eval()
    with torch.no_grad():
      x = _to_tensor(state)
      return self.main_q_network(x).argmax().item()

  def memorize(self, state, action, reward, next_state, done):
    """遷移を Replay バッファに追加する。"""
    self.memory.push(state, action, reward, next_state, done)

  def update_main_q_network(self):
    """
    Replay からミニバッチをサンプルし、Q ネットワークを更新する。
    TD 目標の max Q(s', a') も同じネットワークで計算（ターゲットネットワークなし）。
    終端遷移では次状態の Q を使わない。
    """
    if len(self.memory) < BATCH_SIZE:
      return

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)
    device = state_batch.device
    done_batch = done_batch.to(device)

    # 現在の Q 値: Q(s, a)。勾配を流して更新する
    q_sa = self.main_q_network(state_batch).gather(1, action_batch).squeeze(1)

    # 次状態の最大 Q 値（同じネットワーク、勾配は流さない）
    with torch.no_grad():
      q_next = self.main_q_network(next_state_batch).max(dim=1)[0]
    # 終端遷移（done=1）では次状態の Q を使わない（TD 目標 = reward のみ）
    q_next_masked = q_next * (1.0 - done_batch)

    # TD 目標: r + γ * (1 - done) * max_a' Q(s', a')
    td_target = reward_batch + GAMMA * q_next_masked

    self.main_q_network.train()
    loss = F.smooth_l1_loss(q_sa, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
