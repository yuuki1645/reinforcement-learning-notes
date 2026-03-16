"""
Q ネットワーク: 状態 → 各行動の Q 値（スカラー）を出力する 3 層 MLP。

CartPole では n_in=4, n_mid=32, n_out=2。
Main Q と Target Q の両方で同じ構造を使う。
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  """状態を入力し、各行動の Q 値を出力する全結合 3 層ネットワーク。"""

  def __init__(self, n_in, n_mid, n_out):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_in, n_mid)
    self.fc2 = nn.Linear(n_mid, n_mid)
    # Dueling Network
    self.fc3_adv = nn.Linear(n_mid, n_out)
    self.fc3_v = nn.Linear(n_mid, 1)

  def forward(self, x):
    """x: (batch, n_in) → 出力: (batch, n_out)"""
    h1 = F.relu(self.fc1(x))
    h2 = F.relu(self.fc2(h1))
    
    adv = self.fc3_adv(h2)
    val = self.fc3_v(h2).expand(-1, adv.size(1))

    output = val + adv - adv.mean(dim=1, keepdim=True).expand(-1, adv.size(1))

    return output