# type: ignore
"""
Double DQN の学習ループ（CartPole）。

終了条件は main の実装のとおり:
- terminated: ポールが倒れた or カートが画面外 → 報酬 -1, done
- step == MAX_STEPS - 1: 最大ステップまで生存 → 報酬 +1, done
- それ以外: 報酬 0, not done
"""

import gymnasium as gym
from agent import Agent


HUMAN_RENDER_MODE = False
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 1000
MAX_STEPS = 200


def calc_reward(terminated, truncated, step):
  """
  報酬と「この遷移でエピソード終了か」を返す。
  - terminated: ポールが±13度以上 or カートが画面外 → -1, done
  - step == MAX_STEPS - 1: 最大ステップまで生存 → +1, done
  - それ以外 → 0, not done
  """
  if terminated:
    return -1.0, True
  if step == MAX_STEPS - 1:
    return 1.0, True
  return 0.0, False


def main():
  env = gym.make(ENV_NAME, render_mode="human" if HUMAN_RENDER_MODE else None)
  agent = Agent(
    num_states=env.observation_space.shape[0],
    num_actions=env.action_space.n,
  )

  for episode in range(NUM_EPISODES):
    state, info = env.reset()

    for step in range(MAX_STEPS):
      action = agent.get_action(state, episode)
      next_state, _reward, terminated, truncated, info = env.step(action)

      reward, done = calc_reward(terminated, truncated, step)

      # next_state は常に env の返り値のまま保存（done でも最後の観測を入れる）
      agent.memorize(state, action, reward, next_state, done)
      agent.update_main_q_network()

      state = next_state

      if done:
        print(f"Episode {episode+1}/{NUM_EPISODES} finished after {step+1}/{MAX_STEPS} steps")
        break

  env.close()


if __name__ == "__main__":
  main()
