"""
Double DQN エージェント。

- 行動選択・TD 目標の「次状態で最大 Q を取る行動」は Main Q で決定
- その行動の Q 値は Target Q で評価（過大評価を抑える）
- 終端（done）では次状態の Q を使わない
- 一定回数 update ごとに Target を Main に同期
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from net import Net
from replay_memory import ReplayMemory


BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE_INTERVAL = 10  # この回数だけ update するごとに Target を Main に同期


def _to_tensor(x, dtype=torch.float32):
  return torch.tensor(x, dtype=dtype)


class Agent:
  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions
    self.memory = ReplayMemory(capacity=10000)

    n_in, n_mid, n_out = num_states, 32, num_actions
    self.main_q_network = Net(n_in, n_mid, n_out)
    self.target_q_network = Net(n_in, n_mid, n_out)
    self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)
    self._n_updates = 0

  def get_action(self, state, episode):
    """ε-greedy で行動を選択する。"""
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
    Replay からミニバッチをサンプルし、Double DQN で Main Q を更新する。
    一定間隔で Target Q を Main に同期する。
    """
    if len(self.memory) < BATCH_SIZE:
      return

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)
    device = state_batch.device
    done_batch = done_batch.to(device)

    # Q(s, a): Main で計算（更新対象）
    q_sa = self.main_q_network(state_batch).gather(1, action_batch).squeeze(1)

    # Double DQN: 次状態で「最大 Q を取る行動」は Main で決め、その Q 値は Target で取る
    with torch.no_grad():
      best_actions_next = self.main_q_network(next_state_batch).argmax(dim=1, keepdim=True)
      q_next = self.target_q_network(next_state_batch).gather(1, best_actions_next).squeeze(1)
    # 終端遷移では次状態の Q を使わない
    q_next_masked = q_next * (1.0 - done_batch)

    td_target = reward_batch + GAMMA * q_next_masked

    self.main_q_network.train()
    loss = F.smooth_l1_loss(q_sa, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self._n_updates += 1
    if self._n_updates % TARGET_UPDATE_INTERVAL == 0:
      self.target_q_network.load_state_dict(self.main_q_network.state_dict())
