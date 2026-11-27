"""第 4 步：写一个【可信的评测函数】—— 后面所有实验都靠它判断"有没有进步"。

为什么需要它：
    训练过程中每 50 回合算一次的"成功率"是【滑动窗口 + 小样本 + 训练用的 seed】，
    噪声极大（实测：500 回合里总共只成功过 1 次，窗口偶尔显示 2%，然后归零）。
    这种数字不能用来判断"学没学会"。

正确做法（本文件）：
    训练前、训练后，各用【同一组固定的、训练没见过的 seed】跑 N 个回合，比较成功率。

跑法：
    cd ~/Documents/mani_skill
    .venv/bin/python my_code/step4_evaluate.py
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

OBS_DIM, ACT_DIM = 42, 8
EVAL_SEEDS = range(10001, 10021)      # ⭐ 评测专用 seed：和训练用的 1..500 完全不重叠


class Policy(nn.Module):
    """和 step3 里同一个结构（复制过来，方便本文件独立运行）。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, ACT_DIM),
        )
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.2))

    def forward(self, obs):
        return self.net(obs)


def evaluate(policy, seeds=EVAL_SEEDS, deterministic=True, verbose=True):
    """在固定 seed 集上评测策略，返回 dict(成功率/平均回报/平均步数)。

    deterministic=True  → 动作取 μ（不采样），结果稳定、方差小
    deterministic=False → 按 σ 采样，反映策略真实的随机性
    """
    env = CPUGymWrapper(gym.make("PickCube-v1", num_envs=1, obs_mode="state"))
    was_training = policy.training
    policy.eval()                                  # 关掉 dropout/BN 之类（本例没有，但养成习惯）

    successes, returns, lengths = [], [], []
    with torch.no_grad():                          # ⭐ 评测不需要梯度，省内存也更快
        for i, seed in enumerate(seeds):
            obs, info = env.reset(seed=seed)
            total_return, steps, success, done = 0.0, 0, False, False

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                mu = policy(obs_t)
                if deterministic:
                    action = mu                   # 确定性：直接取均值
                else:
                    std = policy.log_std.exp().expand_as(mu)
                    action = torch.distributions.Normal(mu, std).sample()
                action = torch.clamp(action, -1.0, 1.0)

                obs, reward, terminated, truncated, info = env.step(action.numpy())
                total_return += float(reward)
                steps += 1
                success = success or bool(info.get("success", False))
                done = terminated or truncated

            successes.append(float(success))
            returns.append(total_return)
            lengths.append(steps)

            if verbose:
                print(f"  评测回合 {i+1:2d} (seed={seed}) | 回报 {total_return:6.2f} "
                      f"| 步数 {steps:3d} | success={success}")

    env.close()
    if was_training:
        policy.train()

    result = {
        "成功率": float(np.mean(successes)),
        "平均回报": float(np.mean(returns)),
        "回报标准差": float(np.std(returns)),
        "平均步数": float(np.mean(lengths)),
        "回合数": len(successes),
    }
    if verbose:
        print(f"\n  ── 汇总（{result['回合数']} 回合）──")
        print(f"     成功率     : {result['成功率']*100:.1f}%")
        print(f"     平均回报   : {result['平均回报']:.2f} ± {result['回报标准差']:.2f}")
        print(f"     平均步数   : {result['平均步数']:.1f}")
    return result


def main():
    policy = Policy()                        # 未训练的网络（随机初始化）
    print("评测【未训练】的策略：")
    r0 = evaluate(policy)
    print("\n⭐ 这就是你的『训练前基线』。等 step5 训练完，用同一个 evaluate() 再评一次，")
    print("   两次的差别才是『学没学会』的可信证据（而不是训练日志里的滑动窗口）。")


if __name__ == "__main__":
    main()
