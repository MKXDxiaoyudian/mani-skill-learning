import gymnasium as gym
import mani_skill.envs
 
#创建仿真环境
env = gym.make(
    "PickCube-v1",                       # 环境名称
    num_envs = 1 ,                       # 并行环境数量，CPU默认是1
    obs_mode = "state" ,                 # 观测模式：状态向量(其他可选“rgb”彩色图像“depth”深度图“segmentation”分割掩码)
    control_mode = "pd_ee_delta_pose" ,  # 控制模式：控制机械臂末端执行器位姿增量（连续动作，7 维）
    render_mode = "human"                # 渲染模式：可视化显示
    )

print("\n--------------------------------------------------")
#输出观测空间与动作空间信息
print("observation_space:", env.observation_space,"\n")
print("action_space:", env.action_space)

print("---------------------------------------------------\n")


#重置环境开始新回合
obs, _= env.reset(seed = 0)#确保实验环境的初始状态是确定且可以重复的
done = False


#主交互循环
while not done:
    action = env.action_space.sample()     #随机采样动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()                           #渲染当前画面


#释放资源
env.close()
