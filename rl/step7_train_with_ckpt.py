"""第 7 步：带「早停 + 最佳 checkpoint」的 PPO 训练。

为什么需要（本轮实验的教训）：
    单环境 PPO 会出现"策略塌缩"——训练越久反而越差（实测 4.56 → 0.45）。
    而且塌缩【不可预测】（同配置另一次跑到 6.57 都没塌）。
    → 所以必须：训练过程中定期评测，保存"评测回报最高"的那一版。

产出：
    my_code/checkpoints/best_policy.pt   最佳模型的参数
    my_code/checkpoints/last_policy.pt   最后一版的参数（对照用）
    my_code/checkpoints/train_log.csv    训练曲线数据（供画图）

跑法：
    cd ~/Documents/mani_skill
    .venv/bin/python my_code/step7_train_with_ckpt.py
"""
import copy
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from step4_evaluate import evaluate

OBS_DIM, ACT_DIM = 42, 8
GAMMA, LAM, LR = 0.99, 0.95, 1e-4
CLIP_EPS, PPO_EPOCHS, MINIBATCH = 0.2, 4, 64
VALUE_COEF, ENTROPY_COEF = 0.5, 0.005
ROLLOUT_EPISODES = 4
TOTAL_ITERS = 500            # 500 × 4 = 2000 回合
EVAL_EVERY = 50              # 每 50 迭代（200 回合）评测一次
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh())
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
        return self.ac(obs)[0]


def collect(env, model, n, seed_base):
    trajs = []
    for k in range(n):
        obs, info = env.reset(seed=seed_base + k)
        tr = {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": []}
        success, done = False, False
        while not done:
            o = torch.as_tensor(obs, dtype=torch.float32)
            dist, v = model.dist(o)
            a = torch.clamp(dist.sample(), -1.0, 1.0)
            tr["obs"].append(o); tr["actions"].append(a.detach())
            tr["log_probs"].append(dist.log_prob(a).sum().detach())
            tr["values"].append(v.detach())
            obs, r, term, trunc, info = env.step(a.detach().numpy())
            tr["rewards"].append(float(r))
            success = success or bool(info.get("success", False))
            done = term or trunc
        tr["success"] = float(success)
        trajs.append(tr)
    return trajs


def compute_gae(rewards, values):
    T = len(rewards); adv = [0.0] * T; gae = 0.0
    for t in reversed(range(T)):
        nv = 0.0 if t == T - 1 else values[t + 1].item()
        delta = rewards[t] + GAMMA * nv - values[t].item()
        gae = delta + GAMMA * LAM * gae
        adv[t] = gae
    return adv, [adv[t] + values[t].item() for t in range(T)]


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    env = CPUGymWrapper(gym.make("PickCube-v1", num_envs=1, obs_mode="state"))
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    good_state = copy.deepcopy(model.state_dict())

    r0 = evaluate(PolicyView(model), verbose=False)
    print(f"[起始] 评测回报 {r0['平均回报']:.2f} | 成功率 {r0['成功率']*100:.1f}%", flush=True)

    best_ret = r0["平均回报"]
    best_ep = 0
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_policy.pt"))
    log_rows = [{"episode": 0, "train_return": 0.0, "eval_return": r0["平均回报"],
                 "success_rate": r0["成功率"], "entropy": float("nan"), "log_std": -1.2}]
    recent_returns, recent_success = [], []

    for it in range(1, TOTAL_ITERS + 1):
        trajs = collect(env, model, ROLLOUT_EPISODES, seed_base=it * 1000)
        ob, ab, lb, advb, retb = [], [], [], [], []
        for tr in trajs:
            adv, ret = compute_gae(tr["rewards"], tr["values"])
            ob.extend(tr["obs"]); ab.extend(tr["actions"])
            lb.extend(tr["log_probs"]); advb.extend(adv); retb.extend(ret)
            recent_returns.append(sum(tr["rewards"])); recent_success.append(tr["success"])
        ob = torch.stack(ob); ab = torch.stack(ab); lb = torch.stack(lb)
        advb = torch.tensor(advb, dtype=torch.float32)
        retb = torch.tensor(retb, dtype=torch.float32)
        advb = (advb - advb.mean()) / (advb.std() + 1e-5)

        N = ob.shape[0]
        ent_val = float("nan")
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(N)
            for st in range(0, N, MINIBATCH):
                idx = perm[st:st + MINIBATCH]
                dist, v = model.dist(ob[idx])
                lp = dist.log_prob(ab[idx]).sum(dim=-1)
                ratio = (lp - lb[idx]).exp()
                s1 = ratio * advb[idx]
                s2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advb[idx]
                pl = -torch.min(s1, s2).mean()
                vl = VALUE_COEF * (v - retb[idx]).pow(2).mean()
                ent = dist.entropy().sum(dim=-1).mean()
                loss = pl + vl - ENTROPY_COEF * ent
                if not torch.isfinite(loss):
                    opt.zero_grad(); continue
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                model.log_std.data.clamp_(LOG_STD_MIN, LOG_STD_MAX)
                ent_val = ent.item()

        # ---- 定期评测 + 保存最佳 ----
        if it % EVAL_EVERY == 0:
            ep = it * ROLLOUT_EPISODES
            r = evaluate(PolicyView(model), verbose=False)
            log_rows.append({"episode": ep,
                             "train_return": float(np.mean(recent_returns[-EVAL_EVERY * ROLLOUT_EPISODES:])),
                             "eval_return": r["平均回报"],
                             "success_rate": r["成功率"],
                             "entropy": ent_val,
                             "log_std": float(model.log_std.mean())})
            flag = ""
            if r["平均回报"] > best_ret:
                best_ret = r["平均回报"]; best_ep = ep
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_policy.pt"))
                flag = "  ← 新最佳，已保存"
            print(f"[回合 {ep:5d}] 训练回报 {np.mean(recent_returns[-EVAL_EVERY*ROLLOUT_EPISODES:]):5.2f} "
                  f"| 评测回报 {r['平均回报']:5.2f} | 成功率 {r['成功率']*100:4.1f}% "
                  f"| 熵 {ent_val:4.2f}{flag}", flush=True)

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last_policy.pt"))
    env.close()

    with open(os.path.join(CKPT_DIR, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "train_return", "eval_return",
                                          "success_rate", "entropy", "log_std"])
        w.writeheader()
        for row in log_rows:
            w.writerow(row)

    # ---- 加载最佳模型，重新评测 ----
    print()
    print("=" * 66, flush=True)
    best_model = ActorCritic()
    best_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_policy.pt")))
    r_best = evaluate(PolicyView(best_model), verbose=False)
    last_model = ActorCritic()
    last_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "last_policy.pt")))
    r_last = evaluate(PolicyView(last_model), verbose=False)

    print(f"最佳 checkpoint（第 {best_ep} 回合）: 评测回报 {r_best['平均回报']:.2f} "
          f"| 成功率 {r_best['成功率']*100:.1f}%")
    print(f"最后 checkpoint（第 {TOTAL_ITERS*ROLLOUT_EPISODES} 回合）: 评测回报 {r_last['平均回报']:.2f} "
          f"| 成功率 {r_last['成功率']*100:.1f}%")
    print(f"起始                                   : 评测回报 {r0['平均回报']:.2f}")
    print(f"→ 早停带来的提升: {r_best['平均回报'] - r_last['平均回报']:+.2f}（相对最后一版）")
    print("=" * 66)


if __name__ == "__main__":
    main()
