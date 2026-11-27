"""第 3 步：REINFORCE —— 第一次真正训练 PickCube 的抓取动作。

和 step_2_policy.py 的区别：
    step2 只做「前向 + 采样 + 算 log_prob」（不训练）
    本文件补上「三块拼图」：① 收集 log_prob/reward  ② 倒推算回报 G_t  ③ 算 loss + 反向传播更新参数

跑法：
    cd ~/Documents/mani_skill
    .venv/bin/python my_code/step3_reinforce.py

⚠️ 预期结果（我实测过）：平均回报会从 2.x 涨到 4.x，但 **成功率可能一直是 0%**。
   这不是 bug —— REINFORCE 单环境方差太大，学不会这个任务；它正是我们后面加 baseline / 上 PPO 的理由。
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

# ---------------- 超参数（先都别动，跑通再说）----------------
OBS_DIM, ACT_DIM = 42, 8     # 来自 step1 打印的 Box 形状
GAMMA = 0.99                 # 折扣因子：越远的未来奖励打折越多
LR = 3e-4                    # 学习率
NUM_EPISODES = 500           # 跑多少个回合（约 1 分钟）
LOG_EVERY = 50               # 每多少回合打印一次
# ----------------------------------------------------------


class Policy(nn.Module):
    """策略网络：输入 42 维观测 → 输出 8 维动作的均值；标准差是独立可学参数。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, ACT_DIM),          # 输出「动作均值」μ
        )
        # ⭐ log_std 必须是 nn.Parameter，优化器才能更新它（否则 σ 永远不变）
        # exp(-1.2) ≈ 0.30，和 step2 里固定的 0.3 对应
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.2))

    def forward(self, obs):
        return self.net(obs)

    def dist(self, obs):
        """返回动作的概率分布：Normal(μ, σ)。"""
        mu = self.net(obs)
        std = self.log_std.exp().expand_as(mu)   # 保证形状和 μ 一致
        return torch.distributions.Normal(mu, std)


def compute_returns(rewards, gamma=GAMMA):
    """从回合末尾往前倒推每一步的回报 G_t = r_t + γ·G_{t+1}。

    为什么要倒推：G_t 依赖 G_{t+1}，从后往前一次遍历即可，不用重复累加。
    """
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()                             # 倒推出来是反序，翻回来对齐时间步
    return torch.tensor(returns, dtype=torch.float32)


def main():
    env = CPUGymWrapper(gym.make("PickCube-v1", num_envs=1, obs_mode="state"))
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    recent_returns, recent_success = [], []

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=ep)            # 每回合换一个种子，增加多样性
        log_probs, rewards, success = [], [], False
        done = False

        # ---------- ① 收集：跑完一个完整回合 ----------
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)   # numpy → float32 张量
            dist = policy.dist(obs_t)
            action = torch.clamp(dist.sample(), -1.0, 1.0)      # 采样并限制在动作空间内
            log_probs.append(dist.log_prob(action).sum())        # 8 维求和 = 整个动作的 log 概率

            obs, reward, terminated, truncated, info = env.step(action.detach().numpy())
            rewards.append(float(reward))
            success = success or bool(info.get("success", False))
            done = terminated or truncated

        # ---------- ② 算回报 + ③ 更新 ----------
        G = compute_returns(rewards)                        # 每一步的 G_t
        loss = -(torch.stack(log_probs) * G).sum()          # REINFORCE：好动作提高概率
        opt.zero_grad()                                     # ① 清掉上回合梯度
        loss.backward()                                     # ② 求梯度
        opt.step()                                          # ③ 更新参数

        recent_returns.append(sum(rewards))
        recent_success.append(float(success))

        if ep % LOG_EVERY == 0:
            print(f"回合 {ep:4d} | 平均回报 {np.mean(recent_returns[-LOG_EVERY:]):6.2f} "
                  f"| 成功率 {np.mean(recent_success[-LOG_EVERY:])*100:5.1f}% "
                  f"| loss {loss.item():8.3f}")

    env.close()
    print(f"\n最终：近 {LOG_EVERY} 回合成功率 {np.mean(recent_success[-LOG_EVERY:])*100:.1f}%")
    print("如果成功率是 0%：不是写错了，是 REINFORCE 单环境学不动 → 下一步加 baseline / 上 PPO")


if __name__ == "__main__":
    main()
