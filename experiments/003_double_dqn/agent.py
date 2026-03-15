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


# 1回の更新で使う遷移数
BATCH_SIZE = 32
# 割引率: 将来の報酬をどれだけ重視するか (0〜1)
GAMMA = 0.99
# この回数だけ update するごとに Target Q の重みを Main Q にコピーする
TARGET_UPDATE_INTERVAL = 10


def _to_tensor(x, dtype=torch.float32):
  """numpy 配列などを PyTorch テンソルに変換（dtype を統一）"""
  return torch.tensor(x, dtype=dtype)


class Agent:
  """Double DQN エージェント（Main Q + Target Q + Experience Replay）"""

  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions
    self.memory = ReplayMemory(capacity=10000)

    # Q ネットワークを2つ: Main（更新対象）と Target（TD 目標用、一定間隔で Main をコピー）
    n_in, n_mid, n_out = num_states, 32, num_actions
    self.main_q_network = Net(n_in, n_mid, n_out)
    self.target_q_network = Net(n_in, n_mid, n_out)
    self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)
    self._n_updates = 0  # update 回数（ターゲット同期のタイミング用）

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
    Replay からミニバッチをサンプルし、Double DQN で Main Q を更新する。
    TD 目標の「次状態の最大 Q」は、Main で行動を決め・Target で Q を評価する。
    一定間隔で Target の重みを Main にコピーする。
    """
    if len(self.memory) < BATCH_SIZE:
      return

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)
    device = state_batch.device
    done_batch = done_batch.to(device)

    # 現在の Q 値: Q(s, a)。Main で計算し、勾配を流して更新する
    q_sa = self.main_q_network(state_batch).gather(1, action_batch).squeeze(1)

    # Double DQN: 次状態 s' で「最大 Q を取る行動」は Main で決め、その行動の Q 値は Target で取る
    # （通常の DQN だと max Q(s',a') が過大評価されやすいため、行動と評価を分離）
    with torch.no_grad():
      best_actions_next = self.main_q_network(next_state_batch).argmax(dim=1, keepdim=True)
      q_next = self.target_q_network(next_state_batch).gather(1, best_actions_next).squeeze(1)
    # 終端遷移（done=1）では次状態の Q を使わない（TD 目標 = reward のみ）
    q_next_masked = q_next * (1.0 - done_batch)

    # TD 目標: r + γ * (1 - done) * Q_target(s', a')  where a' = argmax Main(s')
    td_target = reward_batch + GAMMA * q_next_masked

    self.main_q_network.train()
    loss = F.smooth_l1_loss(q_sa, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self._n_updates += 1
    if self._n_updates % TARGET_UPDATE_INTERVAL == 0:
      self.target_q_network.load_state_dict(self.main_q_network.state_dict())
