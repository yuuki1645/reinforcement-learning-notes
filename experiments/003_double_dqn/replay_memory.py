"""
Experience Replay 用のバッファ。

遷移 (state, action, reward, next_state, done) を貯め、
ランダムにサンプリングしてミニバッチ学習に使う。
next_state は常に観測配列（done でも env が返す最後の状態を保存する）。
"""

from collections import namedtuple
import random

import numpy as np
import torch


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
  """固定容量の Replay バッファ。古い遷移は上書きされる（リングバッファ）。"""

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.index = 0

  def push(self, state, action, reward, next_state, done):
    """遷移を 1 件追加する。満杯なら最も古いものを上書き。"""
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.index] = Transition(state, action, reward, next_state, done)
    self.index = (self.index + 1) % self.capacity

  def __len__(self):
    return len(self.memory)

  def sample(self, batch_size):
    """
    バッファからランダムに batch_size 件サンプルし、
    すべて (batch_size, ...) の固定長テンソルで返す。
    """
    samples = random.sample(self.memory, batch_size)
    states = np.array([s.state for s in samples], dtype=np.float32)
    actions = np.array([s.action for s in samples], dtype=np.int64)
    rewards = np.array([s.reward for s in samples], dtype=np.float32)
    next_states = np.array([s.next_state for s in samples], dtype=np.float32)
    dones = np.array([s.done for s in samples], dtype=np.float32)
    return (
      torch.tensor(states),
      torch.tensor(actions).unsqueeze(1),
      torch.tensor(rewards),
      torch.tensor(next_states),
      torch.tensor(dones),
    )
