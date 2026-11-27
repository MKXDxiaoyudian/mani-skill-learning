"""单回合运行 my_pick_cube_env 里的自定义环境（书 3.3 结构的 PickCube 版）。

用法：
    cd ~/Documents/mani_skill
    .venv/bin/python my_code/run_my_pick_cube_env.py
"""
import gymnasium as gym

import mani_skill.envs              # 注册 ManiSkill 内置任务
import my_pick_cube_env             # ⭐ 必须 import，@register_env 才会执行

TASK_ID = "MyPickCube-v1"           # 必须和 @register_env 的名字一致

if TASK_ID not in gym.registry:
    raise SystemExit(
        f"❌ 找不到 {TASK_ID}\n"
        f"   检查 my_code/my_pick_cube_env.py 里 @register_env 的名字是否等于 \"{TASK_ID}\"。"
    )

env = gym.make(TASK_ID, num_envs=1, obs_mode="state", render_mode="human")
print("observation_space:", env.observation_space)
print("action_space:     ", env.action_space)

obs, info = env.reset(seed=0)
viewer = env.render()

done, steps = False, 0
try:
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = bool(terminated or truncated)
        steps += 1
        viewer = env.render()
finally:
    print(f"回合结束：{steps} 步｜info={info}")
    env.close()
