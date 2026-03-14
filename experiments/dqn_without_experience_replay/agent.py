import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



GAMMA = 0.99



class Agent:
  def __init__(self, num_states, num_actions):
    self.num_actions = num_actions

    # Qネットワークの定義
    self.model = nn.Sequential()
    self.model.add_module("fc1", nn.Linear(num_states, 32))
    self.model.add_module("relu1", nn.ReLU())
    self.model.add_module("fc2", nn.Linear(32, 32))
    self.model.add_module("relu2", nn.ReLU())
    self.model.add_module("fc3", nn.Linear(32, num_actions))

    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

  def get_action(self, state, episode):
    epsilon = 0.5 * (1 / (episode + 1)) # エピソードが進むにつれてepsilonを小さくする

    if np.random.uniform(0, 1) < epsilon:
      return np.random.randint(0, self.num_actions)
    else:
      self.model.eval() # ネットワークを推論モードにする
      with torch.no_grad(): # 勾配計算を無視する
        return self.model(torch.tensor(state)).argmax().item()

  def update(self, state, action, reward, next_state):
    self.model.train() # ネットワークを訓練モードにする

    # 状態stateで行動actionを選んだときのQ値を取得
    state_action_value = self.model(torch.tensor(state))[action]
    # 教師信号を計算する
    expected_state_action_value = torch.tensor(reward + GAMMA * self.model(torch.tensor(next_state)).max().item())

    # 損失関数を計算する
    loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)

    # ネットワークのパラメータを更新する
    self.optimizer.zero_grad() # 勾配をリセットする
    loss.backward() # 勾配を計算する
    self.optimizer.step() # パラメータを更新する
