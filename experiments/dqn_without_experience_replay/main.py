# type: ignore

import gymnasium as gym
from agent import Agent



HUMAN_RENDER_MODE = True
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 1000
MAX_STEPS = 200



def calc_reward(terminated, truncated, step):
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
    num_actions=env.action_space.n
  )

  complete_episodes = 0 # 連続成功回数

  for episode in range(NUM_EPISODES):
    state, info = env.reset()
    
    for step in range(MAX_STEPS):
      action = agent.get_action(state, episode)

      # env.stepが返す_rewardは使わない
      next_state, _reward, terminated, truncated, info = env.step(action)

      reward = calc_reward(terminated, truncated, step)

      # Qネットワークの更新
      agent.update(state, action, reward, next_state)

      state = next_state

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



if __name__ == "__main__":
  main()