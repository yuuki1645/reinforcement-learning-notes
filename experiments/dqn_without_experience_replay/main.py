# type: ignore

"""
DQN（Experience Replay なし）の学習ループ。

CartPole を最大 MAX_STEPS ステップまで行い、報酬はシェイピングで与える。
10エピソード連続で成功（step >= 196 で終了）したら学習を打ち切る。
"""

import gymnasium as gym
from agent import Agent


# 描画するか（False にすると学習が速い）
HUMAN_RENDER_MODE = False
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 1000
# 1エピソードの最大ステップ数（これに達したら「成功」とみなす）
MAX_STEPS = 200


def calc_reward(terminated, truncated, step):
  """
  報酬のシェイピング。
  - 長く立てていれば +1（step > 195 のとき）
  - 途中で倒れ or 制限に達したら -1
  - それ以外は 0
  """
  if step > 195:
    return 1.0
  elif terminated or truncated:
    return -1.0
  else:
    return 0.0


def main():
  env = gym.make(ENV_NAME, render_mode="human" if HUMAN_RENDER_MODE else None)
  agent = Agent(
    num_states=env.observation_space.shape[0],
    num_actions=env.action_space.n,
  )

  # 連続で「成功」したエピソード数（10回続いたら終了）
  complete_episodes = 0

  for episode in range(NUM_EPISODES):
    state, info = env.reset()

    for step in range(MAX_STEPS):
      action = agent.get_action(state, episode)

      next_state, _reward, terminated, truncated, info = env.step(action)
      # 環境の報酬は使わず、シェイピングした報酬で学習
      reward = calc_reward(terminated, truncated, step)

      # 遷移を1回だけ使って更新（Experience Replay なし）
      agent.update(state, action, reward, next_state)

      state = next_state

      # ほぼ最後まで立てていたら「成功」
      if step >= 196:
        complete_episodes += 1
        print(f"Episode {episode+1}/{NUM_EPISODES} Clear! | Complete episodes: {complete_episodes}")
        break

      if terminated or truncated:
        complete_episodes = 0
        print(f"Episode {episode+1}/{NUM_EPISODES} finished after {step+1}/{MAX_STEPS} steps")
        break

    if complete_episodes >= 10:
      print(f"10回連続成功！ エピソード数: {episode+1}")
      break

  env.close()


if __name__ == "__main__":
  main()