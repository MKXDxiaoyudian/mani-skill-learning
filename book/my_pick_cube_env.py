"""我的自定义环境：用 PickCube 复现书 3.3 的「环境结构」。

对照关系（书 3.3 的 OpenStorageDrawerEnv → 这里的 MyPickCubeEnv）：
    书 3.3                          →  这份文件
    @register_env("...", 200)        →  @register_env("MyPickCube-v1", 50)
    class ...(BaseEnv)               →  class MyPickCubeEnv(PickCubeEnv)   ← 继承现成任务，不用从零
    def __init__(...): ...           →  不用重写（PickCubeEnv 已有）
    def _default_sim_config          →  不用重写
    def _load_agent(self, options)   →  ⭐ 只覆盖这一个（书里手打的就是它）
    _load_scene / _initialize_episode / evaluate → 全部沿用 PickCube 的实现

为什么这样做：书 3.3 的抽屉任务依赖 storage1.ucsd.edu 的资产，那个服务器已挂
（上游 issue #1489，至今未修），所以用资产随包、不需要下载的 PickCube 来跑通同一套结构。
"""
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env


@register_env("MyPickCube-v1", max_episode_steps=50)
class MyPickCubeEnv(PickCubeEnv):
    def _load_agent(self, options: dict):
        # ⭐ 这就是书 3.3 里手打的 _load_agent，只是位置换成 PickCube 桌面的位置。
        #   p = 机器人基座在世界坐标系里的位置 [x, y, z]
        #   q = 基座朝向的四元数 —— ⚠️ SAPIEN 是 [w, x, y, z]（w 在第一位！不是 xyzw）
        #       q=[1,0,0,0] = 不旋转（单位旋转）
        #       q=[0,0,0,1] = 绕 Z 轴转 180°（机器人会背对桌子）→ 可以把下面这行换成它对比看
        # ⚠️ 注意：不能写 super()._load_agent(...)，因为 PickCubeEnv._load_agent(self, options)
        #   只收 options 一个参数；要传位姿必须直接调 BaseEnv 的那一版。
        BaseEnv._load_agent(self, options, sapien.Pose(p=[-0.615, 0, 0], q=[1, 0, 0, 0]))
