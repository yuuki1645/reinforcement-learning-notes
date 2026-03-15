# type: ignore
import gymnasium as gym
from agent import Agent



HUMAN_RENDER_MODE = False
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 1000
MAX_STEPS = 200



def calc_reward(terminated, truncated, step):
  """
  CartPole-v1でtruncatedがTrueになるのは、stepが499以上になった時。
  今回はMAX_STEPSが200なので、truncatedがTrueになることはない。
  ---
  terminatedがTrueになるのは、poleが±13度以上倒れた時か、
  cartが画面外に出た時。
  ---
  戻り値は、報酬とdoneかどうか。
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
    state, info = env.reset() # s_0

    for step in range(MAX_STEPS):
      # ε-greedy で行動を選択
      action = agent.get_action(state, episode)

      next_state, _reward, terminated, truncated, info = env.step(action)

      reward, done = calc_reward(terminated, truncated, step)
      
      if done:
        next_state = None
      
      # print(f"step: {step}, done: {done}, next_state: {next_state}")

      agent.memorize(state, action, reward, next_state, done)

      # DDQN & Experience Replay
      agent.update_main_q_network()

      state = next_state

      if done:
        print(f"Episode {episode+1}/{NUM_EPISODES} finished after {step+1}/{MAX_STEPS} steps")
        
        if episode % 2 == 0:
          agent.update_target_q_network()
        
        break


if __name__ == "__main__":
  main()