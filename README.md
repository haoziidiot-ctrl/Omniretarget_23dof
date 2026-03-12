# Omniretarget_23dof

Omniretarget_23dof框架是在原Holosoma框架上改成23dof的g1版本，是一个全面的人形机器人框架，用于在人形机器人上训练和部署强化学习策略，以及进行动作重定向（motion retargeting）。 它支持跨多个模拟器（IsaacGym、IsaacSim、MJWarp、MuJoCo）的移动（速度跟踪）和全身跟踪任务，并支持 PPO 和 FastSAC 等算法。

本项目将结合GVHMR视频提取，GMR机器人重定向，以及Beyondmimic人形机器人运动跟踪框架进行完整的一个tracking motion的完整训练部署流程。
Omniretarget更多起重定向功能，用于将复杂高难度或与物体接触动作进行重定向，实现高精度接触要求。

本项目提供将GVHMR的pt格式转换到npz格式（在holosoma中）
```
python src/holosoma_retargeting/tools/convert_hmr4d_results_pt_to_smplx_npz.py \
  --pt src/holosoma_retargeting/demo_data/dance/zuixuan_minfoot.pt \
  --out src/holosoma_retargeting/demo_data/dance/zuixuan_minfoot.npz \
  --smpl-model-dir {models_smplx_v1_1/models} \
  --model-type smplx \
  --swap-yz --flip-y \
  --height-axis z --height-from joints \
  --batch-size 256
```
其中，
	--pt pt文件地址
	--out 生成npz文件地址
	--smpl-model-dir 从[SMPLX]官网下载的smplx模型(https://smpl-x.is.tue.mpg.de/)

## 重定向步骤
##### 将smplx格式的npz文件重定向到指定23dof机器人上
```
python examples/robot_retarget.py \
  --task-type robot_only \
  --task-name zuixuan_minfoot \
  --data_path demo_data/dance \
  --data_format smplx \
  --robot g1_23dof \
  --save_dir demo_results/g1_23dof/robot_only/dance

由于**[Byondmimic_23dof](https://gitlab.i.rokae.com/huanghao/beyondmimic_23dof.git)** 可以实现训练框架
并已与**[Robomimic_23dof](https://gitlab.i.rokae.com/huanghao/robomimic_deploy_23dof.git)**的部署框架打通，故不在次项目中进行策略训练


以下是原作者仓库其他内容以及我使用调整过的训练框架
## 特征
- **多模拟器支持**：IsaacGym、IsaacSim、MuJoCo Warp (MJWarp) 和 MuJoCo（仅用于推理）。
- **多种强化学习算法**：PPO 和 FastSAC。
- **机器人支持**：Unitree G1 和 Booster T1 人形机器人。
- **任务类型**：移动（速度跟踪）和全身跟踪。
- **Sim-to-sim 和 sim-to-real 部署**：在仿真和真实机器人控制之间共享推理流水线。
- **动作重定向**：将人类动作捕捉数据转换为机器人动作，同时保留与物体和地形的交互。
- **Wandb 集成**：视频记录、自动上传 ONNX 权重文件（checkpoint），以及直接从 Wandb 加载权重文件。

## 仓库结构

```text
src/
├── holosoma/              # 核心训练框架（移动与全身跟踪）
├── holosoma_inference/    # 推理与部署流水线
└── holosoma_retargeting/  # 从人类动作数据到机器人的动作重定向
```

## 文档

- **[训练指南](src/holosoma/README.md)** - 在 IsaacGym/IsaacSim 中训练移动和全身跟踪策略。
- **[推理与部署指南](src/holosoma_inference/README.md)** - 将策略部署到真实机器人或在 MuJoCo 仿真中进行评估。
- **[重定向指南](src/holosoma_retargeting/README.md)** - 将人类动作捕捉数据转换为机器人动作。

## 快速开始

### 设置

根据您的用例选择合适的设置脚本：

```bash
# 用于 IsaacGym 训练
bash scripts/setup_isaacgym.sh

# 用于 IsaacSim 训练
# 由于 IsaacSim 的依赖关系，需要 Ubuntu 22.04 或更高版本
bash scripts/setup_isaacsim.sh

# 用于 MJWarp 训练和 MuJoCo 仿真（推理）
bash scripts/setup_mujoco.sh

# 用于推理/部署
bash scripts/setup_inference.sh

# 用于动作重定向
bash scripts/setup_retargeting.sh
```

### 训练步骤

以lafan数据集为例（命令中具体动作的motion file以及保存路径自行调整）
#### 1.LAFAN 上的单序列重定向
```
python examples/robot_retarget.py \
--data_path demo_data/lafan \
--task-type robot_only \
--task-name dance1_subject1 \
--data_format lafan \
--task-config.ground-range -10 10 \
--save_dir demo_results/g1/robot_only/lafan \
--retargeter.debug \
--retargeter.visualize \
--retargeter.foot-sticking-tolerance 0.02
```
#### 2.训练前可视化动作
```
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1 \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/my_box_carrying.npz"      
```
#### 3.将重定向后的 `.npz` 文件转换为强化学习训练所需的格式
```
xvfb-run -a python data_conversion/convert_data_format_mj.py \
    --input_file ./demo_results/g1_23dof/robot_only/dance/daoma_grounded.npz \
    --output_fps 50 \
    --output_name converted_res/robot_only/daoma_23dof_omni.npz \
    --data_format smplx \
    --object_name "ground" \
    --once
```
#### 4.训练自定义动作文件
```
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --algo.config.save-interval=200 \
    --logger.video.enabled False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/daoma.npz"
```
##### 继续训练
```
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --logger.resume True \
    --algo.config.save-interval=200 \
    --logger.video.enabled False \
    --training.checkpoint logs/WholeBodyTracking/20260114_033032-g1_29dof_wbt_manager-locomotion/model_13400.pt \
    --command.setup_terms.motion_command.params.motion_config.motion_file="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/dance1_subject1_mj_fps50.npz"
```
##### 多 GPU 训练
```
source scripts/source_isaacgym_setup.sh
torchrun --nproc_per_node=4 src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --algo.config.save-interval=200 \
    --logger.video.enabled=False \
    --training.num-envs=16384 \
    --training.checkpoint logs/WholeBodyTracking/20260104_102247-g1_29dof_wbt_manager-locomotion/model_08000.pt \
    --command.setup_terms.motion_command.params.motion_config.motion_file="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/dance2_subject1_mj_fps50.npz"
```

> **注意：** 对于无头服务器（headless servers），请参阅[训练指南](src/holosoma/README.md#video-recording)以获取视频记录配置说明。

有关更多示例和配置选项，请参阅[训练指南](src/holosoma/README.md)。

### 快速演示

我们提供了运行完整流水线的脚本：（针对 LAFAN 数据集的下载和处理）、重定向、数据转换以及全身跟踪策略训练。

```bash
# 使用 OMOMO 数据运行重定向和全身跟踪策略训练
bash demo_scripts/demo_omomo_wb_tracking.sh

# 使用 LAFAN 数据运行重定向和全身跟踪策略训练
bash demo_scripts/demo_lafan_wb_tracking.sh
```

