# type: ignore

# a2c_cartpole.py
# Python 3.10+
# pip install torch gymnasium

import math
from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# =========================================================
# 1. ハイパーパラメータ
# =========================================================
@dataclass
class Config:
    env_name: str = "CartPole-v1"
    hidden_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99

    # ★ A2C は「何ステップ分まとめて学習するか」が重要
    rollout_steps: int = 5

    # 学習全体の設定
    max_episodes: int = 500
    max_steps_per_episode: int = 1000

    # 損失の重み
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # 表示
    print_interval: int = 10

    # 再現性
    seed: int = 42


# =========================================================
# 2. Actor-Critic ネットワーク
# =========================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()

        # ★ 共通の特徴抽出部分
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
        )

        # Actor: 行動の確率 logits を出す
        self.policy_head = nn.Linear(hidden_size, action_dim)

        # Critic: 状態価値 V(s) を出す
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.policy_head(h)          # shape: [batch, action_dim]
        value = self.value_head(h)            # shape: [batch, 1]
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor):
        
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# =========================================================
# 3. ロールアウト用バッファ
# =========================================================
class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.entropies = []

    def add(self, obs, action, log_prob, reward, done, value, entropy):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.entropies.append(entropy)

    def clear(self):
        self.__init__()


# =========================================================
# 4. n-step target と Advantage を計算
# =========================================================
def compute_returns_and_advantages(
    rewards: list[float],
    dones: list[bool],
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
):
    """
    rewards: 長さ T の list
    dones:   長さ T の list
    values:  shape [T]
    next_value: shape [1] もしくはスカラー tensor

    返り値:
        returns:     shape [T]
        advantages:  shape [T]
    """
    T = len(rewards)
    returns = torch.zeros(T, dtype=torch.float32)

    # ★ 後ろから計算する
    # done=True のとき、その先の価値は打ち切る
    running_return = next_value.squeeze().detach()

    for t in reversed(range(T)):
        if dones[t]:
            running_return = torch.tensor(0.0)

        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    # Advantage = target_value - current_value
    advantages = returns - values.detach()
    return returns, advantages


# =========================================================
# 5. 学習1回分
# =========================================================
def update_model(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    next_value: torch.Tensor,
    cfg: Config,
):
    # list -> tensor
    log_probs = torch.stack(buffer.log_probs)              # [T]
    values = torch.stack(buffer.values).squeeze(-1)        # [T]
    entropies = torch.stack(buffer.entropies)              # [T]

    returns, advantages = compute_returns_and_advantages(
        rewards=buffer.rewards,
        dones=buffer.dones,
        values=values,
        next_value=next_value,
        gamma=cfg.gamma,
    )

    # Actor loss
    # ★ Advantage が正なら、その行動の確率を上げる
    actor_loss = -(log_probs * advantages).mean()

    # Critic loss
    # ★ V(s) が target_value に近づくように学習
    critic_loss = (returns - values).pow(2).mean()

    # Entropy bonus
    # ★ 方策が偏りすぎるのを少し防ぐ
    entropy_bonus = entropies.mean()

    total_loss = (
        actor_loss
        + cfg.value_loss_coef * critic_loss
        - cfg.entropy_coef * entropy_bonus
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": entropy_bonus.item(),
    }


# =========================================================
# 6. メイン学習ループ
# =========================================================
def train():
    cfg = Config()

    # シード固定
    torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(obs_dim, action_dim, cfg.hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    episode_rewards = []

    for episode in range(1, cfg.max_episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0
        step_count = 0

        buffer = RolloutBuffer()

        while not (done or truncated):
            step_count += 1

            # obs -> tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            # print(f"obs_tensor: {obs_tensor}")
            # print(f"obs_tensor.shape: {obs_tensor.shape}")
            # exit()

            # 行動選択
            action, log_prob, entropy, value = model.get_action_and_value(obs_tensor)

            # 環境を1ステップ進める
            next_obs, reward, done, truncated, _ = env.step(action.item())

            # バッファに保存
            buffer.add(
                obs=obs_tensor,
                action=action,
                log_prob=log_prob.squeeze(0),
                reward=reward,
                done=(done or truncated),
                value=value.squeeze(0),
                entropy=entropy.squeeze(0),
            )

            obs = next_obs
            episode_reward += reward

            # rollout_steps ごと、またはエピソード終了時に更新
            if len(buffer.rewards) == cfg.rollout_steps or done or truncated:
                with torch.no_grad():
                    if done or truncated:
                        next_value = torch.tensor([0.0], dtype=torch.float32)
                    else:
                        next_obs_tensor = torch.tensor(
                            obs, dtype=torch.float32
                        ).unsqueeze(0)
                        _, next_value_raw = model(next_obs_tensor)
                        next_value = next_value_raw.squeeze(0)

                update_info = update_model(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    next_value=next_value,
                    cfg=cfg,
                )

                buffer.clear()

            if step_count >= cfg.max_steps_per_episode:
                break

        episode_rewards.append(episode_reward)

        if episode % cfg.print_interval == 0:
            recent_avg = sum(episode_rewards[-cfg.print_interval:]) / cfg.print_interval
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:6.1f} | "
                f"Recent Avg: {recent_avg:6.1f} | "
                f"Actor Loss: {update_info['actor_loss']:.4f} | "
                f"Critic Loss: {update_info['critic_loss']:.4f} | "
                f"Entropy: {update_info['entropy']:.4f}"
            )

    env.close()
    print("Training finished.")


# =========================================================
# 7. 実行
# =========================================================
if __name__ == "__main__":
    train()