import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from net import Net
from replay_memory import ReplayMemory



BATCH_SIZE = 32
GAMMA = 0.99



def _to_tensor(x, dtype=torch.float32):
  return torch.tensor(x, dtype=dtype)



class Agent:
  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions

    self.memory = ReplayMemory(capacity=10000)

    # ニューラルネットワークを構築(main & target)
    n_in, n_mid, n_out = num_states, 32, num_actions
    self.main_q_network = Net(n_in, n_mid, n_out)
    self.target_q_network = Net(n_in, n_mid, n_out)

    self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

  def get_action(self, state, episode):
    """
    ε-greedy で行動を選択する。
    エピソードが進むほど ε を小さくし、探索から活用へ移行する。
    """
    epsilon = 0.5 * (1 / (episode + 1))

    if np.random.uniform(0, 1) < epsilon:
      return np.random.randint(0, self.num_actions)
    else:
      self.main_q_network.eval()
      with torch.no_grad():
        action = self.main_q_network(_to_tensor(state)).argmax().item()
        return action

  def memorize(self, state, action, reward, next_state, done):
    self.memory.push(state, action, reward, next_state, done)

  def update_main_q_network(self):
    if len(self.memory) < BATCH_SIZE:
      return

    state_batch, action_batch, reward_batch, non_final_next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)

    self.main_q_network.eval()

    state_action_values = self.main_q_network(state_batch).gather(dim=1, index=action_batch)

    # 次の状態での最大Q値の行動a_mをMain Q-Networkから求める
    a_m = self.main_q_network(non_final_next_state_batch).max(dim=1)[1].view(-1, 1)

    next_state_action_values = torch.zeros(BATCH_SIZE)

    non_final_mask = torch.tensor(tuple(map(lambda d: d is False, done_batch)), dtype=torch.bool)

    # 次の状態があるindexの行動a_mのQ値をTarget Q-Networkから求める
    next_state_action_values[non_final_mask] = self.target_q_network(non_final_next_state_batch).gather(dim=1, index=a_m).squeeze().detach()

    expected_state_action_values = reward_batch + GAMMA * next_state_action_values

    self.main_q_network.train()

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # 結合パラメータを更新する
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def update_target_q_network(self):
    self.target_q_network.load_state_dict(self.main_q_network.state_dict())