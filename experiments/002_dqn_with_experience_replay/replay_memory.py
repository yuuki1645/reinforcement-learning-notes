from collections import namedtuple
import random
import torch



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))



def _to_tensor(x, dtype=torch.float32):
  return torch.tensor(x, dtype=dtype)



class ReplayMemory:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.index = 0

  def push(self, state, action, reward, next_state):
    if len(self.memory) < self.capacity:
      self.memory.append(None) # メモリが満タンでないときはNoneを追加

    self.memory[self.index] = Transition(state, action, reward, next_state)
    self.index = (self.index + 1) % self.capacity

  def __len__(self):
    return len(self.memory)

  def sample(self, batch_size):
    samples = random.sample(self.memory, batch_size)

    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    
    for sample in samples:
      state_batch.append(sample.state)
      action_batch.append(sample.action)
      reward_batch.append(sample.reward)
      next_state_batch.append(sample.next_state)

    return \
      _to_tensor(state_batch), \
      _to_tensor(action_batch, dtype=torch.long).unsqueeze(1), \
      _to_tensor(reward_batch), \
      _to_tensor(next_state_batch)