import numpy as np



TD_ERROR_EPSILON = 0.0001



class TDerrorMemory:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.index = 0

  def push(self, td_error):
    if len(self.memory) < self.capacity:
      self.memory.append(None)

    self.memory[self.index] = td_error
    self.index = (self.index + 1) % self.capacity

  def __len__(self):
    return len(self.memory)

  def get_prioritized_indexes(self, batch_size):
    # TD誤差の和を計算
    sum_absolute_td_error = np.sum(np.absolute(self.memory))
    sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory) # 微小値を足す

    # batch_size分の乱数を生成して、昇順に並べる
    rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
    rand_list = np.sort(rand_list)

    indexes = []
    idx = 0
    tmp_sum_absolute_td_error = 0
    for rand_num in rand_list:
      while tmp_sum_absolute_td_error < rand_num:
        tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
        idx += 1

      if idx >= len(self.memory):
        idx = len(self.memory) - 1
      indexes.append(idx)

    return indexes

  def update_td_error(self, updated_td_errors):
    self.memory = updated_td_errors

    
