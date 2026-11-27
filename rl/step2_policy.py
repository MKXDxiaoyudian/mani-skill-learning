import torch 
import torch.nn as nn
import gymnasium as gym
import  mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

OBS_DIM , ACT_DIM = 42, 8


class Policy(nn.Module):
    def __init__(self):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, ACT_DIM),

        )
        self.std = torch.full((ACT_DIM,),0.3)

    def forward(self, obs):
        mu = self.net(obs)
        return mu

def main():
    env = CPUGymWrapper(gym.make("PickCube-v1", num_envs = 1, obs_mode = "state"))
    obs, info = env.reset(seed = 0)

    policy = Policy()
    obs_t = torch.as_tensor(obs, dtype = torch.float32)

    mu = policy(obs_t)
    print("mu(8维动作均值)：", mu.detach().numpy().round(3))
    print("std:", policy.std)

    for k in range (3):
        dist = torch.distributions.Normal(mu, policy.std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        logp = dist.log_prob(action).sum()
        print(f"采样{k+1}：action = {action.detach().numpy().round(3)} log_prob = {logp.item():.3f}")
    env.close ()

if __name__ == "__main__" :
    main()