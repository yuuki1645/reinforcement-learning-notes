# type: ignore

"""
DQN（Experience Replay なし）の学習ループ。

CartPole を最大 MAX_STEPS ステップまで行い、報酬はシェイピングで与える。
10エピソード連続で成功（step >= 196 で終了）したら学習を打ち切る。

使い方:
  python main.py <出力JSONのパス>
  python main.py results/run001.json
"""

import argparse
import json
from pathlib import Path

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
  parser = argparse.ArgumentParser(description="DQN (Experience Replay なし) で CartPole を学習し、結果をJSONに出力する")
  parser.add_argument(
    "output_path",
    type=Path,
    help="実験結果を書き出すJSONファイルのパス（例: results/run001.json）",
  )
  args = parser.parse_args()
  output_path: Path = args.output_path

  env = gym.make(ENV_NAME, render_mode="human" if HUMAN_RENDER_MODE else None)
  agent = Agent(
    num_states=env.observation_space.shape[0],
    num_actions=env.action_space.n,
  )

  # 各エピソードの結果を記録
  episodes = []
  # 10回連続成功に達したときのエピソード番号（1始まり）。未達なら None
  episode_at_10_consecutive_success = None
  # 連続で「成功」したエピソード数（10回続いたら終了）
  complete_episodes = 0

  for episode in range(NUM_EPISODES):
    state, info = env.reset()
    steps_in_episode = 0  # このエピソードで何ステップ進んだか

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
        steps_in_episode = step + 1
        print(f"Episode {episode+1}/{NUM_EPISODES} Clear! | Complete episodes: {complete_episodes}")
        break

      if terminated or truncated:
        complete_episodes = 0
        steps_in_episode = step + 1
        print(f"Episode {episode+1}/{NUM_EPISODES} finished after {step+1}/{MAX_STEPS} steps")
        break

    episodes.append({
      "episode_index": episode,
      "steps": steps_in_episode,
      "success": steps_in_episode >= 196,
    })

    if complete_episodes >= 10:
      episode_at_10_consecutive_success = episode + 1  # 1始まりで記録
      print(f"10回連続成功！ エピソード数: {episode_at_10_consecutive_success}")
      break

  env.close()

  # 出力: 10連続成功時のエピソード数 + 各エピソードの内容
  output_data = {
    "episode_at_10_consecutive_success": episode_at_10_consecutive_success,
    "episodes": episodes,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
  print(f"結果を保存しました: {output_path}")


if __name__ == "__main__":
  main()