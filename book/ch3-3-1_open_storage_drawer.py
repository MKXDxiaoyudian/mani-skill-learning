# pathlib.Path：用对象方式拼路径（比字符串拼接安全）
from pathlib import Path
# typing：给变量标类型，运行时不影响，只帮你和 IDE 看懂
from typing import Any, Dict, Union 
# numpy：数值计算（这里主要用来做数组/常量）
import numpy as np
# sapien：底层物理仿真引擎（ManiSkill 建在它之上）；Pose 是它的「位置+朝向」类型
import sapien
# torch：ManiSkill 的批量/设备接口都用 torch 张量（为了以后能上 GPU 并行）
import torch

# Panda：Franka Panda 机械臂这个「智能体」的类实现
#   任务里的机器人不是写死的，而是通过 robot_uids="panda" 去注册表里取类
from mani_skill.agents.robots import Panda
# BaseEnv：ManiSkill 所有任务的基类。继承它 = 自动获得
#   reset/step/render/观测构建/控制器/并行环境 等全套能力
from mani_skill.envs.sapien_env import BaseEnv
# CameraConfig：描述一台「相机」的配置（分辨率、视场角、安装位姿）
#   注意它只描述，不在这里创建相机；由 BaseEnv 按配置去建
from mani_skill.sensors.camera import CameraConfig
# common：通用小工具；sapien_utils：常用位姿构造（例如 look_at 相机看向某点）
from mani_skill.utils import common, sapien_utils
# actors：往场景里加「静态/简单形体」（球、盒子、地面等）的工厂函数
from mani_skill.utils.building import actors
# build_ground：专门造一块地平面（所有任务几乎都要）
from mani_skill.utils.building.ground import build_ground
# register_env：把类注册成 gym 环境名的装饰器
#   ✅ 已修正为 register_env（原来写成 rengister_env）
from mani_skill.utils.registration import register_env
# structs：Articulation（有关节的物体，如柜子/机械臂）、Link（连杆）、Pose（位姿）
#   Pose 来自 sapien，这里再导入一次是为了类型标注方便
from mani_skill.utils.structs import Articulation, Link, Pose
# types：SimConfig（仿真频率/GPU 内存等）、GPUMemoryConfig（GPU 仿真显存上限）
#   ✅ 已修正为 GPUMemoryConfig（原来写成 GPUMeroryConfig）
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# ---------- 资产路径 ----------
#urdf路径
# 书里这里是省略号：你需要真的储物柜 URDF 文件路径
#   ⚠️ 目前是 ...（Ellipsis），下面 loader.parse 会直接失败
URDF_PATH = ...

# 注册环境：字符串就是 gym.make() 时用的名字；max_episode_steps=200 → 一个回最多 200 步
#   ✅ 装饰器名同上，已修正
@register_env("OpenStorageDrawer-v1", max_episode_steps = 200)
# 自定义任务 = 继承 BaseEnv；类名随意，注册名才是对外接口
class OpenStorageDrawerEnv(BaseEnv):
# 允许用哪些机器人（传别的会被警告）
    SUPPORTED_ROBOTS = ["panda"]
# 类型标注：这个任务的 agent 是 Panda 类型（只给 IDE/类型检查看）
    agent: Union[Panda]
# 成功判据的参数：抽屉当前位移 ≥ 最大行程的 30% 就算成功（书 3.3 原文）
    min_open_frac: float = 0.3 #成功所需的最小开度比例

# 构造函数：ManiSkill 创建环境时会调它
#   *args 往父类透传（num_envs / obs_mode / render_mode 等都走这里）
    def __init__(
# 关键字参数开始
        self,
# *args：把位置参数原样转给父类
        *args,
# 默认机器人是 panda；改这里就换机器人（要同步改 SUPPORTED_ROBOTS）
#   ✅ 行尾英文逗号已补上（原来缺逗号 → SyntaxError）
        robot_uids: str = "panda",
# 初始关节位置的随机噪声幅度（±0.02 rad）：每回合起始姿态略有不同 → 策略更鲁棒
        robot_init_qpos_noise: float = 0.02, #初始化随机噪声，以增强泛化能力
# **kwargs：其余未知参数一并转给父类，避免漏传
        **kwargs,

# 构造函数签名结束
    ) :
# 存下来，供 _initialize_episode 里生成噪声用
#   ✅ 属性名已修正为 self.robot_init_qpos_noise（原来写成 robot_init_qposInoise）
        self.robot_init_qpos_noise = robot_init_qpos_noise
# ⭐ 关键一步：调父类 BaseEnv.__init__，它负责真正干重活：
#   建场景/建 agent/套控制器/按 render_mode 开 viewer（所以必须 super 先/后调用一次）
        super().__init__(*args, robot_uids= robot_uids, **kwargs)


    # ---------- 仿真参数配置（书 3.3 的「2. 仿真与传感器配置」）----------
    # _default_sim_config 是「属性」而不是方法：BaseEnv 在初始化时会读取它
    #仿真与传感器设置 _default_sim_config属性设置了接触点和碰撞面的最大数量
    # @property：当作属性访问（self._default_sim_config），不用加括号调用
    @property
    # 返回 SimConfig：仿真频率、控制频率、GPU 显存、场景参数都在这里
    #   ✅ 方法名已修正为 _default_sim_config（原来写成 _defuault_sim_config）
    def _default_sim_config(self):
    # SimConfig(...)：只写了 GPU 显存这一项，其余用默认（sim 100Hz / control 20Hz）
        return SimConfig(
    # GPU 显存上限配置；只在「GPU 仿真 + 多环境并行」时才有意义
    #   单环境（num_envs=1）时 ManiSkill 自动用 CPU 仿真，这一项基本用不到
    #   ✅ 已修正为 GPUMemoryConfig
            gpu_memory_config = GPUMemoryConfig(
    # 同时存在的「刚体-刚体接触点」上限 2^20 = 1,048,576 个
            max_rigid_contact_count = 2**20, #接触点最大数量
    # 同时存在的「碰撞面片(patch)」上限 2^18 = 262,144 个
            max_rigid_patch_count = 2**18, # 碰撞面最大数量
            )
        )


    # ---------- 传感器（相机）配置 ----------
    @property
    # _default_sensor_configs：任务「感知用」的相机（观测里可能读它的图像）
    def _default_sensor_configs(self):
    # sapien_utils.look_at(eye=相机位置, target=看向的目标点) → 算出相机 Pose（含朝向）
    #   这是最常用的建相机方式：你只给「站哪、看哪」
        pose = sapien_utils.look_at(eye = [1.2,0.8,1.0], target = [0.0,0.0,0.4])
    # CameraConfig(名字, 位姿, 宽, 高, fov)：fov=np.pi/3 就是 60° 视场角
    #   ⚠️ 两个问题：① 书里这个是 128×128，这里宽 128 高 512 是长方形；② 返回值应是 list（见下行注释）
        return [CameraConfig ("base_camera", pose = pose, width = 128, height = 128, fov =np.pi/3)]#宽128px，高512px，视场角60度

    # _default_human_render_camera_configs：给人看的渲染相机（分辨率高、不影响训练）
    @property
    # 同上，只是位置更高更远、512×512 更清楚（书里说这个用于人工观察/录视频）
    def _default_human_render_camera_configs(self):
    #   ✅ 此处的写法已修正为与 ManiSkill 一致
        pose = sapien_utils.look_at(eye = [1.4, 1.0, 1.2], target = [0.0, 0.0, 0.4])
    # ⚠️ 这两个方法都应该 return [CameraConfig(...)]（列表）；
    #   这里返回的是单个对象，父类遍历时会出错
        return CameraConfig("render_camera", pose = pose, width = 512, height = 512, fov = np.pi/3)


    # ---------- 智能体（机器人）加载 ----------
    #智能体与场景加载
    # _load_agent：决定机器人「出现在哪、朝哪」。options 是创建环境时传的额外配置
    def _load_agent(self, options:dict):
    # ⭐ 四元数顺序是 [w, x, y, z]（SAPIEN 的约定，w 在第一位）
    #   p=[0.95,0,0]：机器人底座在世界坐标 x=+0.95 处
    #   q=[0,0,0,1]：w=0 → 这是「绕 Z 轴转 180°」，不是「不旋转」
    #   转 180° 后机器人正面朝 -X 方向，正好对着放在原点的柜子（所以这个值是故意的）
    #   单位旋转应写成 q=[1,0,0,0]
        super()._load_agent(options, sapien.Pose(p = [0.95,0,0], q = [0,0,0,1]))#放置坐标p[x,y,z], 旋转四元数q[w,x,y,z], 这里将机械臂朝向-X方向正对柜子

    # _load_scene：把「场景里的东西」造出来（地面、柜子、标记球……）
    #   方法名 _load 就是「加载/构建」，ManiSkill 在初始化时自动调用
    def _load_scene(self, options: dict):
    # 先铺地面：没有地面的话重力会让所有东西一直往下掉
        self.ground = build_ground(self.scene)#创建地面

    # create_urdf_loader()：创建一个「URDF 导入器」，专门用来把外部 URDF 读进场景
        loader = self.scene.create_urdf_loader()
    # name：给这个导入器起名（出现在调试信息/场景层级里，方便排查）
        loader.name = "storage_furniture"
    # fix_root_link=True：把 URDF 的根连杆「钉死」在世界上（柜子不能整体移动）
    #   桌面操作类任务一般要固定底座；移动操作任务则设 False
        loader.fix_root_link = True #固定柜体底座

    # parse(路径)：解析 URDF 文件，返回一个 dict，里面装着各类「构建器(builder)」
        parsed = loader.parse(str(URDF_PATH))
    # 从字典里取 articulation builders 的第 0 个：
    #   articulation = 「有关节的一串连杆」，柜子（带抽屉关节）就是一个 articulation
        builder = parsed["articulation_builders"][0]

    # 柜子整体的初始位姿：位置 (0,0,1)、朝向四元数 [0.5,0.5,0.5,0.5]（wxyz 顺序）
    #   这个四元数 = 绕某轴转 120°/90° 的组合，作用是让 URDF 的坐标系对齐到「抽屉朝 +X」
    #   ✅ 已修正为 builder.initial_pose（原来写成 builder.intial_pose）
        builder.initial_pose = sapien.Pose(p = [0, 0, 1], q = [0.5, 0.5, 0.5, 0.5])#旋转矩阵： 绕X旋转90度，再绕Z旋转90度
    # build(name=...)：按上面的设置真正把柜子实体加进场景，并保存引用以备后用
    #   类型标注 Articulation = 「它是一串有 joints 的连杆」
        self.cabinet: Articulation = builder.build(name = "storage_furniture")

    # 下面要找到「抽屉所在的连杆」，因为初始化/奖励都要读它的位移
        #识别移动关节所在连杆
    # 先置 None，找到后再赋值（None 表示「还没有」）
        self.drawer_link: Link = None
    # zip(links, joints)：同时遍历连杆与关节（ManiSkill 保证两者顺序一一对应）
        for link, joint in zip(self.cabinet.links,self.cabinet.joints):
    # joint.type（这里 joint.type 是个列表/数组，取 [0]）若为 prismatic = 平移关节
    #   prismatic（抽屉/滑轨） vs revolute（铰链/门）—— 这是判断「抽屉还是门」的关键
            if joint.type[0] == "prismatic":
    # 命中就记下这条连杆（抽屉本体），后面用它的位移算开度
                self.drawer_link = link
    # 找到第一个就跳出循环（这样一个柜子只取一个抽屉）
                break

    # 造一个绿色半透明小球当「目标位置指示器」，让人眼能看见目标在哪
        #创建可视化标记点
    # actors.build_sphere(...)：往场景里加一个球
    #   radius 半径 0.15m；color=[r,g,b,a]，a=0.8 表示 80% 不透明
        self.drawer_goal_maker = actors.build_sphere(
    # name：实体名（调试用）
            self.scene, radius = 0.15, color = [0, 1, 0, 0.8],
    # body_type="kinematic"：运动学物体——不受力和碰撞影响，「摆哪就在哪」
    #   正好适合当纯视觉标记；add_collision=False：不参与碰撞（不会被机器人撞飞）
            name = "drawer_goal_maker", body_type = "kinematic",
    # initial_pose：先放在原点上方 0.15m 处（真正的每回合位置在 _initialize_episode 里更新）
            add_collision = False, initial_pose = sapien.Pose(p = [0, 0, 0.15]),

    # 调用结束（这个右括号单独一行只是代码风格）
        )
    # _hidden_objects：不需要出现在传感器观测里的物体（标记球不该被相机看见）
    #   ✅ 已修正为 self.drawer_goal_maker（原来多写了一个下划线）
        self._hidden_objects.append(self.drawer_goal_maker)# 从传感器观测中隐藏



    # ---------- 后处理：等所有物体建好后，再做「依赖位姿/物理」的调整 ----------
    # 为什么要有这个钩子：_load_scene 时物体位置还没最终确定，只有建完之后才能读真实位姿
    #后处理配置与物理参数调整
    # _after_reconfigure：每次场景重建后都会被调用一次（重置配置时也可能再调）
    def _after_reconfigure(self, options: dict):
    
    # 【目标】让柜子底部正好贴地：先算「当前柜子底部比地高多少」，之后每回合按这个值平移
        #计算柜体垂直偏移量，使底部贴地
    # get_first_collision_mesh()：取第一个碰撞网格（trimesh 对象），用来算包围盒
        collision_mesh = self.cabinet.get_first_collision_mesh()
    # pose.p 是 (num_envs, 3) 的位置张量；[0, 2] 取第 0 个环境的 z 坐标
        current_z = float (self.cabinet.pose.p[0, 2])
    # bounding_box.bounds 是 [[xmin,ymin,zmin],[xmax,ymax,zmax]]；[0,2] 取 zmin（最低点）
    #   ✅ 已修正为 bounding_box（原来写成 boding_box）
        mesh_z_min = float (collision_mesh.bounding_box.bounds[0, 2])
    # 偏移量 = (柜子当前 z) − (网格最低点 z)，取负号后加上去就能让最低点落到 z=0
    #   即：柜子当前位置 + offset 后，最低点正好在 0（地面）
        self.cabinet_z_offset = float(-(mesh_z_min - current_z))



    # ---------- 物理参数调整：让抽屉更好拉、更稳 ----------
        #调整物理参数
    # 局部导入物理模块（只有用到时才导，能省一点启动时间）
        import sapien.physx as physx
    # PhysxMaterial：定义「材质」= 静摩擦 3.0 / 动摩擦 3.0（数值越大越「涩」）
    #   为什么调大摩擦：抽屉把手要能被夹爪抓住并拉动，摩擦太小会打滑、拉不开
        high_friction_mat = physx.PhysxMaterial(static_friction = 3.0, dynamic_friction = 3.0)
    # 遍历柜子的每一根连杆（每根连杆上有物理组件：质量、碰撞形状……）
        for link in self.cabinet.links:
    # _objs[0]：该连杆在「第 0 个并行环境」里的底层 sapien 对象（私有 API，版本升级可能变）
            obj = link._objs[0]
    # components[0]：该实体上的第 1 个物理组件（通常是 PhysxRigidBody/刚体）
            physx_comp = obj.entity.components[0]
    # 只针对名为 link_2 的连杆单独处理（具体是哪根要看该 URDF 的命名）
            if link.name == "link_2":
    # 把它的质量设成 1.0 kg —— 质量影响惯性：太大会拉不动，太小会乱飞
                physx_comp.mass = 1.0
    # collision_shapes：这根连杆上所有碰撞形状（一个连杆可以有多块碰撞体）
            for shape in physx_comp.collision_shapes:
    # 给每个碰撞形状套上高摩擦材质（前面定义的）
                shape.physical_material = high_friction_mat



    # ---------- 算「成功阈值」：抽屉开到多少算成功 ----------
        #计算成功阈值

    # joint.limits：关节行程限制，形状 (n, 2)，每行 [下限, 上限]
    #   对平移关节来说就是「全关的位置」和「全开的位置」
        qlimits = self.drawer_link.joint.limits 
    # qmin = 下限（抽屉完全闭合时的位置）
        qmin = qlimits[0,0]
    # qmax = 上限（完全打开时的位置）
        qmax = qlimits[0,1]
    # 目标位置 = 下限 + (上限-下限) × 30%  → 开度达到 30% 的位置
    #   unsqueeze(-1) 是「把标量变成 (n,1) 张量」，方便和 (n,1) 的 qpos 比较
    #   ✅ 已修正为 unsqueeze（原来写成 unqueeze）
        self.target_qpos = (qmin + (qmax - qmin) * self.min_open_frac).unsqueeze(-1)



    # ---------- 每回合重置时的初始化（随机化就在这里做）----------
    #回合初始化与随机化
    # _initialize_episode：每次 env.reset() 都会被调用，用来摆好这一回合的初始状态
    #   env_idx：这一批里「哪些并行环境」要重置（单环境时就是 [0]）
    #   ✅ 方法名已修正为 _initialize_episode（原来写成 _iniialize_episode）
    def _initialize_episode(self, env_idx:torch.Tensor, options: dict):
    # torch.device(self.device)：把下面新建的张量放到环境所在设备（CPU 或 GPU）上
    #   设备不一致会报「device mismatch」
        with torch.device(self.device):
    # b = 这一批要重置的环境数（单环境时 = 1）
            b = len(env_idx)



    # ① 摆柜子：让底部贴地（用 _after_reconfigure 里算好的偏移）
            #放置柜体，底部贴地

    # 先造一个 (b, 3) 的全 0 位置张量（x=y=0, z=0）
            cabinet_p = torch.zeros((b,3))
    # 只改 z 列：加上底面偏移量 → 柜子正好坐在地面上
            cabinet_p[:, 2] = self.cabinet_z_offset
    # 朝向四元数 (w,x,y,z)=[0.5,0.5,0.5,0.5]；unsqueeze(0).expand(b,-1) 复制成 b 份，形状 (b,4)
            cabinet_q = torch.tensor([0.5,0.5,0.5,0.5]).unsqueeze(0).expand(b,-1)
    # set_pose 真正把柜子摆到上面算出的位置/朝向（Pose.create_from_pq 用 p+q 造 Pose）
            self.cabinet.set_pose(Pose.create_from_pq(p = cabinet_p, q = cabinet_q))



    # ② 把抽屉关到底：每回合都从「全关」状态开始，保证任务一致
            #关闭抽屉
    # get_qlimits()：读出这个 articulation 所有关节的 [下限,上限]，形状 (num_envs, n_joints, 2)
            qlimits = self.cabinet.get_qlimits()
    # qlimits[env_idx, :, 0] → 取这些环境的第 0 列 = 各关节下限；set_qpos 直接设关节位置
            self.cabinet.set_qpos(qlimits[env_idx, :,0])
    # 同时把关节速度清零（qpos*0 得到全 0 的同形状张量）：否则上一回合的残余速度会带进来
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)



    # ③ 初始化机械臂关节角：给一个「预备姿态」rest_qpos（7 个手臂关节 + 2 个夹爪手指）
            #初始化机械臂关节位置1

    # 书中给的 Panda 站立姿态（弧度）。抄下来即可，不必推导
            rest_qpos = np.array([0.157,0.157,-0.154,-1.99,0.007,3.716,2.377,0.04,0.04])
    # 生成 (b, 9) 的正态噪声：均值 0、标准差 robot_init_qpos_noise
    #   ✅ 已修正为 self._episode_rng.normal(mean, std, size)
            noise = self._episode_rng.normal(0,self.robot_init_qpos_noise,(b,len(rest_qpos)))
    # 初始关节 = rest_qpos + 噪声（每回合略有不同 → 泛化）；unsqueeze(0) 是给 batch 维
            qpos = torch.tensor(rest_qpos, dtype = torch.float32).unsqueeze(0) + torch.tensor(noise)
    # set_qpos：把关节角应用到机器人上
    #   ✅ 已修正为 self.agent.robot（原来写成 robo）
            self.agent.robot.set_qpos(qpos)
    # set_pose：同时把机器人底座摆回固定位置（防止上一回合被撞歪）
            self.agent.robot.set_pose(sapien.Pose(p = [0.95, 0, 0],q = [0, 0, 0, 1]))



    # ④ GPU 仿真才需要的手动「数据同步」：把 CPU 侧的改动用 GPU 缓冲刷一遍
    #   CPU 仿真（num_envs=1）时这段不会执行
    #   ✅ 此处的写法已修正为与 ManiSkill 一致
    #      px.step() 会真的推进一帧物理，正常流程里一般不用在这里调用
            #GPU数据同步
    # 判断当前是否在用 GPU 仿真
            if self.gpu_sim_enabled:
    # 把场景里所有「已修改」的属性上传到 GPU
                self.scene._gpu_apply_all()
    # 更新 GPU 侧 articulation 的运动学量（关节位置对应的连杆位姿）
                self.scene.px.gpu_update_articulation_kinematics()
    # 推进一个物理步（让上面的改动生效）
                self.scene.px.step()
    # 把 GPU 结果取回 CPU，供观测/奖励计算使用
                self.scene._gpu_fetch_all()

    # ⑤ 把绿色标记球挪到抽屉把手的当前位置（告诉人「该抓这里」）
            #更新可视化标记点位置
    # 求把手的当前位置（该函数由你自己实现，书里对应 handle_link_positions 一类的写法）
    #   ⚠️ 目前文件里没有定义 _get_drawer_handle_pos，会 AttributeError；
            drawer_pos = self._get_drawer_handle_pos(env_idx)
    # set_pose 把标记球摆过去（只给 p、不给 q，朝向保持不变）
            self.drawer_goal_maker.set_pose(Pose.create_from_pq(p = drawer_pos))


