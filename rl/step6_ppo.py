"""第 6 步：PPO —— 在 Actor-Critic 基础上加「重要性比率裁剪」和「多轮更新」。

和 step5 的区别（只多了三样东西）：
    ① 不再"一个回合更新一次"，而是【攒几个回合 → 反复训练几遍】
       → 同样数据被利用 PPO_EPOCHS 次，样本效率大增
    ② 用「新旧策略的比率」判断"这次更新改得太多了吗"
           ratio = exp( log_prob_new − log_prob_old )
    ③ 裁剪：ratio 超出 [1−ε, 1+ε] 就截断
           loss = −min( ratio·A,  clip(ratio, 1−ε, 1+ε)·A )

为什么这招有效：
    step5 每更新一次就把数据扔掉；PPO 把数据回收利用，
    但用 clip 拦住"改太多"的步子 —— 这就是它比 A2C 稳的原因。

跑法：
    cd ~/Documents/mani_skill
    .venv/bin/python my_code/step6_ppo.py
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
LAM = 0.95              # GAE 的 λ：偏差/方差折中
LR = 3e-4
CLIP_EPS = 0.2          # PPO 的裁剪范围
PPO_EPOCHS = 4          # 同一批数据重复训练几遍
MINIBATCH = 64
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01     # 熵奖励：防策略过早变得确定
ROLLOUT_EPISODES = 4    # 每次更新前攒几个回合
TOTAL_ITERS = 125       # 125 × 4 回合 = 500 回合（和 step5 同量级，便于对比）
LOG_EVERY = 25


class ActorCritic(nn.Module):
    """与 step5 相同结构：共享主干 + actor 头 + critic 头。"""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(64, ACT_DIM)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.2))

    def forward(self, obs):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def dist(self, obs):
        mu, v = self(obs)
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std), v


class PolicyView(nn.Module):
    """只输出 μ，供 evaluate() 使用。"""

    def __init__(self, ac):
        super().__init__()
        self.ac = ac

    def forward(self, obs):
        mu, _ = self.ac(obs)
        return mu


def collect(env, model, n_episodes, seed_base):
    """跑 n_episodes 个完整回合，返回每回合的轨迹字典。"""
    trajs = []
    for k in range(n_episodes):
        obs, info = env.reset(seed=seed_base + k)
        traj = {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": []}
        success, done = False, False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            dist, v = model.dist(obs_t)
            action = torch.clamp(dist.sample(), -1.0, 1.0)

            traj["obs"].append(obs_t)
            traj["actions"].append(action.detach())
            traj["log_probs"].append(dist.log_prob(action).sum().detach())
            traj["values"].append(v.detach())

            obs, reward, terminated, truncated, info = env.step(action.detach().numpy())
            traj["rewards"].append(float(reward))
            success = success or bool(info.get("success", False))
            done = terminated or truncated
        traj["success"] = float(success)
        trajs.append(traj)
    return trajs


def compute_gae(rewards, values):
    """单回合 GAE（回合末尾 done=True，所以最后一步的 next_value = 0）。"""
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1].item()
        delta = rewards[t] + GAMMA * next_value - values[t].item()
        gae = delta + GAMMA * LAM * gae
        advantages[t] = gae
    returns = [advantages[t] + values[t].item() for t in range(T)]
    return advantages, returns


def main():
    env = CPUGymWrapper(gym.make("PickCube-v1", num_envs=1, obs_mode="state"))
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("=== 训练前评测（固定 seed 10001–10020，确定性动作）===")
    r_before = evaluate(PolicyView(model), verbose=False)
    print(f"    成功率 {r_before['成功率']*100:.1f}%  |  平均回报 {r_before['平均回报']:.2f} "
          f"± {r_before['回报标准差']:.2f}\n")

    recent_returns, recent_success = [], []

    for it in range(1, TOTAL_ITERS + 1):
        # ---------- ① 收集若干回合 ----------
        trajs = collect(env, model, ROLLOUT_EPISODES, seed_base=it * 1000)

        # ---------- ② 拼成一个大 batch，算 GAE ----------
        obs_b, act_b, logp_old_b, adv_b, ret_b = [], [], [], [], []
        for tr in trajs:
            adv, ret = compute_gae(tr["rewards"], tr["values"])
            obs_b.extend(tr["obs"]); act_b.extend(tr["actions"])
            logp_old_b.extend(tr["log_probs"]); adv_b.extend(adv); ret_b.extend(ret)
            recent_returns.append(sum(tr["rewards"])); recent_success.append(tr["success"])

        obs_b = torch.stack(obs_b)                 # (N, 42)
        act_b = torch.stack(act_b)                 # (N, 8)
        logp_old_b = torch.stack(logp_old_b)       # (N,)
        adv_b = torch.tensor(adv_b, dtype=torch.float32)
        ret_b = torch.tensor(ret_b, dtype=torch.float32)

        # 优势标准化：让 Actor 的更新尺度稳定
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # ---------- ③ PPO 多轮更新 ----------
        N = obs_b.shape[0]
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(N)
            for start in range(0, N, MINIBATCH):
                idx = perm[start:start + MINIBATCH]

                dist, v = model.dist(obs_b[idx])
                logp_new = dist.log_prob(act_b[idx]).sum(dim=-1)
                ratio = (logp_new - logp_old_b[idx]).exp()      # ⭐ 新旧策略比率

                surr1 = ratio * adv_b[idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b[idx]
                policy_loss = -torch.min(surr1, surr2).mean()   # ⭐ 裁剪切掉"改太过"的部分
                value_loss = VALUE_COEF * (v - ret_b[idx]).pow(2).mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss + value_loss - ENTROPY_COEF * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # 梯度裁剪，防爆炸
                opt.step()

        if it % LOG_EVERY == 0:
            n = LOG_EVERY * ROLLOUT_EPISODES
            print(f"迭代 {it:3d}（累计 {it*ROLLOUT_EPISODES:3d} 回合）"
                  f" | 平均回报 {np.mean(recent_returns[-n:]):6.2f}"
                  f" | 成功率 {np.mean(recent_success[-n:])*100:5.1f}%"
                  f" | 本块成功 {int(sum(recent_success[-n:]))}"
                  f" | 策略损失 {policy_loss.item():7.2f}"
                  f" | 价值损失 {value_loss.item():6.2f}"
                  f" | 熵 {entropy.item():5.2f}")

    env.close()

    print()
    print("=== 训练后评测（同一组 seed，同一评测函数）===")
    r_after = evaluate(PolicyView(model), verbose=False)
    print(f"    成功率 {r_after['成功率']*100:.1f}%  |  平均回报 {r_after['平均回报']:.2f} "
          f"± {r_after['回报标准差']:.2f}")
    print()
    print("=== 三步对比（同一评测集 seed 10001–10020）===")
    print(f"    成功率  : {r_before['成功率']*100:.1f}%  →  {r_after['成功率']*100:.1f}%")
    print(f"    平均回报: {r_before['平均回报']:.2f}  →  {r_after['平均回报']:.2f}")


if __name__ == "__main__":
    main()
