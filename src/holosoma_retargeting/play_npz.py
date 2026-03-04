import numpy as np
import time
import viser
import argparse
import os
import yourdfpy
from viser.extras import ViserUrdf  # <--- 官方使用的核心工具

def main():
    parser = argparse.ArgumentParser(description="Official-style NPZ Player using ViserUrdf")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the .npz file")
    parser.add_argument("--urdf_path", type=str, default="models/g1/g1_29dof.urdf", help="Path to robot URDF")
    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"Error: NPZ file not found at {args.npz_path}")
        return

    # 1. 加载数据
    print(f"Loading data from {args.npz_path}...")
    data = np.load(args.npz_path)
    
    # 兼容各种 key
    if 'qpos' in data: q_pos = data['qpos']
    elif 'q' in data: q_pos = data['q']
    elif 'robot_q' in data: q_pos = data['robot_q']
    else:
        print("Error: Cannot find qpos/q/robot_q. Keys:", list(data.keys()))
        return

    n_frames, n_dims = q_pos.shape
    print(f"Data shape: {q_pos.shape}")

    # 2. 启动 Viser 服务器
    server = viser.ViserServer(port=8080)
    print("\nVisualizer started at http://localhost:8080")

    # 3. 创建坐标系和加载机器人 (完全模仿 interaction_mesh_retargeter.py 的写法)
    # 创建一个父级坐标系，方便整体移动机器人
    robot_base = server.scene.add_frame("/world/robot", show_axes=False)

    # 加载 URDF
    print(f"Loading URDF from {args.urdf_path}...")
    robot_urdf = yourdfpy.URDF.load(args.urdf_path, load_meshes=True, build_scene_graph=True)

    # 核心：使用 ViserUrdf 自动管理显示
    # 它会自动解析 meshes，你再也不用担心 Mesh vs Scene 的报错了
    viser_robot = ViserUrdf(
        server,
        urdf_or_path=robot_urdf,
        root_node_name="/world/robot"  # 挂载到刚才创建的坐标系下
    )

    # 4. 创建进度条 (仿照官方 visualize_motion)
    # 这里的 slider 允许你在网页上拖动进度
    gui_step = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    
    # 播放控制按钮
    play_button = server.gui.add_button("Play/Pause")
    is_playing = True

    @play_button.on_click
    def _(_):
        nonlocal is_playing
        is_playing = not is_playing

    # 5. 动画循环
    while True:
        frame_idx = gui_step.value
        curr = q_pos[frame_idx]

        # 解析数据
        root_pos = np.zeros(3)
        root_rot = np.array([1, 0, 0, 0]) # w,x,y,z
        joints = np.zeros(29) # G1 default

        if n_dims >= 36: # Root (7) + Joints (29)
            root_pos = curr[0:3]
            root_rot = curr[3:7] 
            joints = curr[7:36]
        elif n_dims == 29: # Only Joints
            joints = curr
        
        # --- 核心更新逻辑 ---
        
        # 1. 更新关节 (ViserUrdf 直接接受数组，不用自己拼字典了！)
        # 注意：这里假设 URDF 里的关节顺序和数据里的顺序是一致的 (通常都是 MuJoCo 顺序)
        viser_robot.update_cfg(joints)

        # 2. 更新基座位置 (通过移动父级坐标系)
        robot_base.position = root_pos
        robot_base.wxyz = root_rot

        # --- 循环控制 ---
        if is_playing:
            next_frame = (frame_idx + 1) % n_frames
            gui_step.value = next_frame
            time.sleep(1/30.0)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    main()