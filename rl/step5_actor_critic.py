"""第 5 步：Actor-Critic —— 给 REINFORCE 加 baseline，让"优势"可正可负。

和 step3 的唯一核心区别（其余几乎一样）：
    step3:  loss = -Σ( log_prob_t × G_t )        ← G_t 全是正数 → 所有动作都被加强
    本文件:  loss = -Σ( log_prob_t × A_t )        ← A_t = G_t − V(s_t)，可正可负
                 A_t > 0 → 这步比预期好 → 加强
                 A_t < 0 → 这步比预期差 → **削弱**（这是关键）

新增的两个部件：
    ① Critic 网络 V(s)：输出"这个状态预期能拿多少分"
    ② 价值损失：让 V(s) 尽量接近实际回报 G_t

数学保证（重要）：减去 baseline 不改变梯度的期望，只降低方差。
    因为 E[ ∇log π · b ] = 0（b 与动作无关时）

跑法：
    cd ~/Documents/mani_skill
    .venv/bin/python my_code/step5_actor_critic.py
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from step4_evaluate import evaluate

OBS_DIM, ACT_DIM = 42, 8
GAMMA = 0.99
LR = 3e-4
NUM_EPISODES = 500
LOG_EVERY = 50
VALUE_COEF = 0.5          # 价值损失在总损失里的权重
ADV_NORM = True           # 是否标准化优势（推荐，能显著稳住训练）


class ActorCritic(nn.Module):
    """共享主干 + 两个头：actor 输出动作均值，critic 输出状态价值。"""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(64, ACT_DIM)      # μ
        self.critic = nn.Linear(64, 1)           # V(s)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.2))

    def forward(self, obs):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def dist(self, obs):
        mu, v = self(obs)
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std), v


class PolicyView(nn.Module):
    """把 ActorCritic 包成"只输出 μ"的策略，好让 step4 的 evaluate() 直接复用。"""

    def __init__(self, ac):
        super().__init__()
        self.ac = ac

    def forward(self, obs):
        mu, _ = self.ac(obs)
        return mu


def compute_returns(rewards, gamma=GAMMA):
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def main():
    env = CPUGymWrapper(gym.make("PickCube-v1", num_envs=1, obs_mode="state"))
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("=== 训练前评测（固定 seed 10001–10020，确定性动作）===")
    r_before = evaluate(PolicyView(model), verbose=False)
    print(f"    成功率 {r_before['成功率']*100:.1f}%  |  平均回报 {r_before['平均回报']:.2f} "
          f"± {r_before['回报标准差']:.2f}")
    print()

    recent_returns, recent_success = [], []

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=ep)
        log_probs, values, rewards = [], [], []
        success, done = False, False

        # ---------- ① 收集：额外记录 V(s) ----------
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            dist, v = model.dist(obs_t)                     # 同时拿分布和价值
            action = torch.clamp(dist.sample(), -1.0, 1.0)
            log_probs.append(dist.log_prob(action).sum())
            values.append(v)                                # ← 新增：记录 V(s_t)

            obs, reward, terminated, truncated, info = env.step(action.detach().numpy())
            rewards.append(float(reward))
            success = success or bool(info.get("success", False))
            done = terminated or truncated

        # ---------- ② 算回报 + 优势 ----------
        G = compute_returns(rewards)                        # 实际回报 G_t
        values = torch.stack(values)                        # 预测的价值 V(s_t)
        advantage = G - values.detach()                     # ⭐ 优势 = 实际 − 预期
        if ADV_NORM and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # ---------- ③ 两个损失 ----------
        policy_loss = -(torch.stack(log_probs) * advantage).sum()
        value_loss = VALUE_COEF * (values - G).pow(2).sum()  # 注意：这里用带梯度的 values
        loss = policy_loss + value_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        recent_returns.append(sum(rewards))
        recent_success.append(float(success))

        if ep % LOG_EVERY == 0:
            print(f"回合 {ep:4d} | 平均回报 {np.mean(recent_returns[-LOG_EVERY:]):6.2f} "
                  f"| 成功率 {np.mean(recent_success[-LOG_EVERY:])*100:5.1f}% "
                  f"| 本块成功 {int(sum(recent_success[-LOG_EVERY:]))} "
                  f"| 策略损失 {policy_loss.item():8.2f} | 价值损失 {value_loss.item():8.2f}")

    env.close()

    print()
    print("=== 训练后评测（同一组 seed，同一评测函数）===")
    r_after = evaluate(PolicyView(model), verbose=False)
    print(f"    成功率 {r_after['成功率']*100:.1f}%  |  平均回报 {r_after['平均回报']:.2f} "
          f"± {r_after['回报标准差']:.2f}")
    print()
    print("=== 对比 ===")
    print(f"    成功率 : {r_before['成功率']*100:.1f}%  →  {r_after['成功率']*100:.1f}%")
    print(f"    平均回报: {r_before['平均回报']:.2f}  →  {r_after['平均回报']:.2f}")


if __name__ == "__main__":
    main()
