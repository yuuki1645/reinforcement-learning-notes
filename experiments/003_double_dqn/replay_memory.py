"""
Experience Replay 用のバッファ。

遷移 (state, action, reward, next_state, done) を貯め、
ランダムにサンプリングしてミニバッチ学習に使う。
done は「その遷移でエピソードが終わったか」を表し、TD 目標の計算で使用する。
"""

from collections import namedtuple
import random

import numpy as np
import torch


# 1 件の遷移を表す型（state, action, reward, next_state, done）
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
  """固定容量の Replay バッファ。古い遷移は上書きされる（リングバッファ）。"""

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.index = 0  # 次に書き込む位置

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
    バッファからランダムに batch_size 件の遷移をサンプルし、
    PyTorch のテンソル（バッチ）として返す。
    np.array で一度まとめてから tensor にすることで変換を速くしている。
    """
    samples = random.sample(self.memory, batch_size)
    states = np.array([s.state for s in samples], dtype=np.float32)
    actions = np.array([s.action for s in samples], dtype=np.int64)
    rewards = np.array([s.reward for s in samples], dtype=np.float32)
    non_final_next_states = np.array([s.next_state for s in samples if s.next_state is not None], dtype=np.float32)
    dones = [s.done for s in samples]
    return (
      torch.tensor(states),
      torch.tensor(actions).unsqueeze(1),  # gather 用に (batch, 1) に
      torch.tensor(rewards),
      torch.tensor(non_final_next_states),
      dones,
    )