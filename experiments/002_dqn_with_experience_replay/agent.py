import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_memory import ReplayMemory



BATCH_SIZE = 32
GAMMA = 0.99



def _to_tensor(x, dtype=torch.float32):
  return torch.tensor(x, dtype=dtype)



class Agent:
  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions

    self.model = nn.Sequential()
    self.model.add_module("fc1", nn.Linear(num_states, 32))
    self.model.add_module("relu1", nn.ReLU())
    self.model.add_module("fc2", nn.Linear(32, 32))
    self.model.add_module("relu2", nn.ReLU())
    self.model.add_module("fc3", nn.Linear(32, num_actions))

    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    self.memory = ReplayMemory(capacity=10000)

  def get_action(self, state, episode):
    epsilon = 0.5 * (1 / (episode + 1))

    if np.random.uniform(0, 1) < epsilon:
      return np.random.randint(0, self.num_actions)
    else:
      self.model.eval()
      with torch.no_grad():
        x = _to_tensor(state)
        q = self.model(x)
        return q.argmax().item()

  def memorize(self, state, action, reward, next_state):
    self.memory.push(state, action, reward, next_state)

  def update(self):
    # Experience Replayでネットワークを更新

    if len(self.memory) < BATCH_SIZE:
      return

    state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample(BATCH_SIZE)
    self.model.eval()

    # ネットワークが出力したQ(s_t, a_t)の値を取得
    state_action_values = self.model(state_batch).gather(dim=1, index=action_batch)

    # print(f"state_action_values: {state_action_values}")

    next_state_action_values = self.model(next_state_batch).max(dim=1)[0].detach()

    # 教師信号
    expected_state_action_values = reward_batch + GAMMA * next_state_action_values

    # print(f"expected_state_action_values: {expected_state_action_values}")

    self.model.train()

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()