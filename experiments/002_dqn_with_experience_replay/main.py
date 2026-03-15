# type: ignore

"""
DQN（Experience Replay あり）の学習ループ。

Replay バッファに遷移を貯め、ランダムにサンプリングしてミニバッチで学習する。
終了条件は 003 と同様:
- terminated: ポールが倒れた or カートが画面外 → 報酬 -1, done
- step == MAX_STEPS - 1: 最大ステップまで生存 → 報酬 +1, done
- それ以外: 報酬 0, not done
10エピソード連続で成功（MAX_STEPS まで生存）したら学習を打ち切り、結果を JSON に保存する。

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
# 1エピソードの最大ステップ数（ここまで生存すれば「成功」）
MAX_STEPS = 200


def calc_reward(terminated, truncated, step):
  """
  報酬と「この遷移でエピソード終了か」を返す（003 と同様）。
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
  # コマンドライン引数: 結果を書き出す JSON のパス
  parser = argparse.ArgumentParser(description="DQN (Experience Replay あり) で CartPole を学習し、結果をJSONに出力する")
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

  # 各エピソードの結果を記録（JSON 出力用）
  episodes = []
  # 10回連続成功に達したときのエピソード番号（1始まり）。未達なら None
  episode_at_10_consecutive_success = None
  # いま何エピソード連続で「成功」しているか
  complete_episodes = 0

  for episode in range(NUM_EPISODES):
    state, info = env.reset()
    steps_in_episode = 0

    for step in range(MAX_STEPS):
      action = agent.get_action(state, episode)
      next_state, _reward, terminated, truncated, info = env.step(action)

      reward, done = calc_reward(terminated, truncated, step)

      agent.memorize(state, action, reward, next_state, done)
      agent.update()

      state = next_state

      if done:
        steps_in_episode = step + 1
        success = steps_in_episode == MAX_STEPS
        if success:
          complete_episodes += 1
          print(f"Episode {episode+1}/{NUM_EPISODES} Clear! | Complete episodes: {complete_episodes}")
        else:
          complete_episodes = 0
          print(f"Episode {episode+1}/{NUM_EPISODES} finished after {steps_in_episode}/{MAX_STEPS} steps")
        break

    # このエピソードの結果を記録（成功 = MAX_STEPS まで生存）
    episodes.append({
      "episode_index": episode,
      "steps": steps_in_episode,
      "success": steps_in_episode == MAX_STEPS,
    })

    if complete_episodes >= 10:
      episode_at_10_consecutive_success = episode + 1
      print(f"10回連続成功！ エピソード数: {episode_at_10_consecutive_success}")
      break

  env.close()

  # 結果を JSON で保存
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