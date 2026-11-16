import gymnasium as gym
import mani_skill.envs
 
#创建仿真环境
env = gym.make(
    "PickCube-v1",                       #环境名称
    num_envs = 1 ,                       #并行环境数量，CPU默认是1
    obs_mode = "state" ,                 #观测模式：状态向量
    control_mode = "pd_ee_delta_pose" ,  # 控制模式：末端执行器位姿增量（连续动作，7 维）
    render_mode = "human"                # 渲染模式：可视化显示
    )

#输出观测空间与动作空间信息
print("observation_space:", env.observation_space)
print("action_space", env.action_space)

#重置环境开始新回合
obs, _= env.reset(seed = 0)
done = False

#主交互循环
while not done:
    action = env.action_space.sample()     #随机采样动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()                           #渲染当前画面


#释放资源
env.close()
