"""单回合运行我改过的 OpenDrawer 环境（书 3.3 复现练习）。

前提：my_code/open_cabinet_drawer.py 里 @register_env 的 uid 是 "MyOpenCabinetDrawer-v1"
（必须换新 uid；用原来的 "OpenCabinetDrawer-v1" 会被内置版占用而静默跳过）。

用法：
    cd ~/Documents/mani_skill
    .venv/bin/python my_code/run_my_drawer.py

这是 1 回合版本：跑满 max_episode_steps（默认 100 步）后窗口自动关闭，属正常现象。
"""
import gymnasium as gym

import mani_skill.envs          # 注册 ManiSkill 内置任务（必须先 import）
import open_cabinet_drawer      # ⭐ 我改过的副本：这一行才让 @register_env 执行

TASK_ID = "MyOpenCabinetDrawer-v1"   # 与你副本里 @register_env 的名字保持一致

"""if TASK_ID not in gym.registry:
    raise SystemExit(
        f"❌ 找不到 {TASK_ID}\n"
        f'   请检查 my_code/open_cabinet_drawer.py 里的 @register_env 名字是否等于 "{TASK_ID}"，\n'
        f"   或者把本文件的 TASK_ID 改成已注册的名字。"
    )"""

env = gym.make(
    TASK_ID, 
    num_envs=1, 
    obs_mode="state", 
    render_mode="human"
    )


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
