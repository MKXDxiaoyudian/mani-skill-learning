import gymnasium as gym
import mani_skill.envs
from  mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

NUM_EPISODE = 3


def make_env ():
    env = gym.make("PickCube-v1", num_envs = 1, obs_mode = "state")
    return CPUGymWrapper(env)


def main():
    env = make_env()
    print("obs_space:", env.observation_space)
    print("act_space:", env.action_space)

    for ep in range(1, NUM_EPISODE + 1):
        obs, info = env.reset(seed = ep)
        total_return, steps, success = 0.0, 0, False
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            steps += 1
            success = success or bool(info.get("success", False))
            if terminated or truncated:
                break

        
        print(f"回合{ep}：总回报 = {total_return:6.2f}  步数 = {steps:3d}  success = {success}")


    env.close()

if __name__ == "__main__":
    main()
