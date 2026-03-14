# type: ignore
import gymnasium as gym
from agent import Agent



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

  episodes = []
  episode_at_10_consecutive_success = None
  complete_episodes = 0

  for episode in range(NUM_EPISODES):
    state, info = env.reset()
    steps_in_episode = 0

    for step in range(MAX_STEPS):
      action = agent.get_action(state, episode)

      next_state, _reward, terminated, truncated, info = env.step(action)

      reward = calc_reward(terminated, truncated, step)

      agent.memorize(state, action, reward, next_state)

      agent.update()

      state = next_state

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
      episode_at_10_consecutive_success = episode + 1
      print(f"10回連続成功！ エピソード数: {episode_at_10_consecutive_success}")
      break

  env.close()




if __name__ == "__main__":
  main()