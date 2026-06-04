# v4_trench_data_collector.py
import os
import h5py
import datetime
import numpy as np
from collections import deque
import cv2
import time
import mujoco
from low_freq_control_v2 import Drone  # 导入新的低层控制器


class DataCollectorV4:
    def __init__(self, drone_instance, data_dir="expert_data_v4", max_buffer_size=50000):
        self.drone = drone_instance
        self.data_dir = data_dir
        self.max_buffer_size = max_buffer_size

        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)

        # 数据缓冲区 - 与v3版本保持相同结构
        self.data_buffer = {
            'images': deque(maxlen=max_buffer_size),
            'states': deque(maxlen=max_buffer_size),
            'velocity_commands_world': deque(maxlen=max_buffer_size),
            'joystick_commands_local': deque(maxlen=max_buffer_size),  # 四维手柄命令
            'timestamps': deque(maxlen=max_buffer_size),
            'positions': deque(maxlen=max_buffer_size),
            'orientations': deque(maxlen=max_buffer_size),
            'velocities': deque(maxlen=max_buffer_size),
            'user_commands': deque(maxlen=max_buffer_size),
            'motor_commands': deque(maxlen=max_buffer_size),
            'mission_status': deque(maxlen=max_buffer_size),
            'raw_joystick_inputs': deque(maxlen=max_buffer_size),
            'target_positions': deque(maxlen=max_buffer_size),  # 新增：目标位置
            'heading_angles': deque(maxlen=max_buffer_size),  # 新增：航向角
            'terrain_params': deque(maxlen=1)  # 新增：地形参数
        }

        # 统计信息
        self.step_count = 0
        self.episode_count = 0
        self.start_time = time.time()

        # 图像参数
        self.image_width = 320
        self.image_height = 240

        # 自动保存设置
        self.save_interval = 3000

        # 标记是否已经保存过数据
        self.data_saved = False

        # 存储当前的速度指令和手柄命令
        self.current_velocity_command = np.zeros(3)
        self.current_joystick_command = np.zeros(4)
        self.current_raw_inputs = np.zeros(4)

        # 存储地形参数
        if hasattr(drone_instance, 'trench_params'):
            self.data_buffer['terrain_params'].append(drone_instance.trench_params)

        print(f"V4数据采集器初始化完成，数据将保存在: {data_dir}")
        print(f"使用新的低层控制器: low_freq_control_v2.Drone")

    def setup_camera(self):
        """设置无人机摄像头用于图像采集"""
        try:
            camera_id = mujoco.mj_name2id(self.drone.m, mujoco.mjtObj.mjOBJ_CAMERA, "drone_eye")
            if camera_id >= 0:
                self.camera_id = camera_id
                print(f"找到无人机摄像头: drone_eye (ID: {camera_id})")
            else:
                # 尝试查找其他摄像头
                for i in range(self.drone.m.ncam):
                    camera_name = mujoco.mj_id2name(self.drone.m, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    if camera_name:
                        print(f"可用摄像头 {i}: {camera_name}")
                self.camera_id = 0
                print("使用默认摄像头")

            self.renderer = mujoco.Renderer(self.drone.m, self.image_height, self.image_width)
            return True

        except Exception as e:
            print(f"摄像头设置失败: {e}")
            return False

    def capture_image(self):
        """从无人机摄像头捕获图像"""
        try:
            self.renderer.update_scene(self.drone.d, camera=self.camera_id)
            image = self.renderer.render()

            if image.shape[:2] != (self.image_height, self.image_width):
                image = cv2.resize(image, (self.image_width, self.image_height))

            # 将图像转换为RGB格式（如果需要）
            if len(image.shape) == 3 and image.shape[2] == 3:
                # MuJoCo渲染器默认返回RGB
                pass
            elif len(image.shape) == 2:
                # 如果是灰度图，转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            return image.astype(np.uint8)

        except Exception as e:
            print(f"图像捕获失败: {e}")
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

    def get_drone_state(self):
        """获取无人机完整状态信息"""
        try:
            se = self.drone.state_estimator

            # 构建状态向量
            state_vector = np.concatenate([
                se.base_pos,  # 3: 位置 (x, y, z)
                se.base_quat,  # 4: 四元数姿态
                se.base_vel_lin_global,  # 3: 线速度 - 世界坐标系
                se.base_vel_ang_local,  # 3: 角速度 - 局部坐标系
                [self.drone.user_cmd.yaw]  # 1: 当前目标偏航角
            ])

            return state_vector
        except Exception as e:
            print(f"获取状态失败: {e}")
            return np.zeros(14)  # 3+4+3+3+1=14维状态向量

    def get_velocity_command(self):
        """获取当前的速度指令 - 基于目标位置计算期望速度"""
        try:
            # 方法1：基于目标位置计算期望速度（与v2控制器逻辑一致）
            target_pos = np.array([self.drone.user_cmd.x, self.drone.user_cmd.y, self.drone.user_cmd.z])
            current_pos = self.drone.state_estimator.base_pos

            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance > 0.1:
                direction_normalized = direction / distance
                # 使用与Drone类相同的速度缩放因子
                desired_speed = min(self.drone.XY_VEL_SCALE, distance * 2.0)
                velocity_command = direction_normalized * desired_speed
            else:
                velocity_command = np.array([0.0, 0.0, 0.0])

            return velocity_command

        except Exception as e:
            print(f"获取速度指令失败: {e}")
            return np.array([0.0, 0.0, 0.0])

    def get_joystick_command_local(self):
        """获取四维手柄控制命令（局部坐标系）- 直接从Drone类获取"""
        try:
            # 直接从Drone类中获取处理后的手柄命令
            if hasattr(self.drone, 'v_x_local_processed') and hasattr(self.drone, 'v_y_local_processed'):
                joystick_cmd = np.array([
                    self.drone.v_x_local_processed,  # x方向局部速度
                    self.drone.v_y_local_processed,  # y方向局部速度
                    self.drone.v_z_processed,  # z方向速度
                    self.drone.yaw_rate_processed  # 期望偏航角速度
                ])
            else:
                # 备选方法：从原始手柄输入计算
                raw_inputs = self.get_raw_joystick_inputs()
                joystick_cmd = np.array([
                    raw_inputs[1] * self.drone.XY_VEL_SCALE,  # v_x_local
                    raw_inputs[0] * self.drone.XY_VEL_SCALE,  # v_y_local
                    raw_inputs[2] * self.drone.Z_VEL_SCALE,  # v_z
                    raw_inputs[3] * self.drone.YAW_RATE_SCALE  # yaw_rate
                ])

            return joystick_cmd

        except Exception as e:
            print(f"获取手柄命令失败: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0])

    def get_raw_joystick_inputs(self):
        """获取原始手柄输入用于调试"""
        try:
            raw_inputs = np.array([0.0, 0.0, 0.0, 0.0])

            if hasattr(self.drone, 'joystick') and self.drone.joystick:
                joystick = self.drone.joystick

                # Xbox控制器轴映射
                # 注意：不同控制器可能有不同的映射，需要根据实际情况调整
                if joystick.get_numbuttons() > 0:
                    # 左摇杆：前后(轴0)和左右(轴1)
                    raw_inputs[0] = joystick.get_axis(0)  # v_y_local (前后)
                    raw_inputs[1] = joystick.get_axis(1)  # v_x_local (左右)
                    raw_inputs[2] = -joystick.get_axis(3) if joystick.get_numbuttons() > 3 else 0.0  # v_z (上下)
                    raw_inputs[3] = -joystick.get_axis(2) if joystick.get_numbuttons() > 2 else 0.0  # yaw_rate (偏航)

            return raw_inputs

        except Exception as e:
            print(f"获取原始手柄输入失败: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0])

    def get_target_position(self):
        """获取当前目标位置"""
        try:
            return np.array([self.drone.user_cmd.x, self.drone.user_cmd.y, self.drone.user_cmd.z])
        except:
            return np.zeros(3)

    def get_heading_angle(self):
        """获取当前航向角"""
        try:
            # 从四元数计算偏航角
            quat = self.drone.state_estimator.base_quat
            w, x, y, z = quat

            # 计算偏航角 (yaw)
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            return yaw
        except:
            return 0.0

    def collect_step_data(self):
        """收集单步数据"""
        try:
            # 捕获图像
            image = self.capture_image()

            # 获取状态信息
            state = self.get_drone_state()

            # 获取当前世界坐标系速度指令
            velocity_command = self.get_velocity_command()

            # 获取四维手柄控制命令（局部坐标系）
            joystick_command = self.get_joystick_command_local()

            # 获取原始手柄输入（用于调试）
            raw_inputs = self.get_raw_joystick_inputs()

            # 获取当前电机指令
            motor_command = self.drone.d.ctrl[:4].copy() if len(self.drone.d.ctrl) >= 4 else np.zeros(4)

            # 获取用户命令
            user_cmd = np.array([
                self.drone.user_cmd.x,
                self.drone.user_cmd.y,
                self.drone.user_cmd.z,
                self.drone.user_cmd.yaw
            ])

            # 获取目标位置和航向角
            target_position = self.get_target_position()
            heading_angle = self.get_heading_angle()

            # 获取时间戳
            timestamp = self.drone.d.time

            # 获取任务状态
            mission_status = 1 if self.drone.mission_complete else 0

            # 存储数据
            self.data_buffer['images'].append(image)
            self.data_buffer['states'].append(state)
            self.data_buffer['velocity_commands_world'].append(velocity_command)
            self.data_buffer['joystick_commands_local'].append(joystick_command)
            self.data_buffer['timestamps'].append(timestamp)
            self.data_buffer['positions'].append(self.drone.state_estimator.base_pos.copy())
            self.data_buffer['orientations'].append(self.drone.state_estimator.base_quat.copy())
            self.data_buffer['velocities'].append(np.concatenate([
                self.drone.state_estimator.base_vel_lin_global,
                self.drone.state_estimator.base_vel_ang_local
            ]))
            self.data_buffer['user_commands'].append(user_cmd)
            self.data_buffer['motor_commands'].append(motor_command)
            self.data_buffer['mission_status'].append(mission_status)
            self.data_buffer['raw_joystick_inputs'].append(raw_inputs)
            self.data_buffer['target_positions'].append(target_position)
            self.data_buffer['heading_angles'].append(heading_angle)

            # 更新当前指令
            self.current_velocity_command = velocity_command
            self.current_joystick_command = joystick_command
            self.current_raw_inputs = raw_inputs

            self.step_count += 1

            # 定期显示采集状态和调试信息
            if self.step_count % 500 == 0:
                self.print_collection_status()

            # 每200步打印一次调试信息
            if self.step_count % 200 == 0:
                print(f"Step {self.step_count}: Pos = [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]")
                print(
                    f"Step {self.step_count}: Vel Cmd = [{velocity_command[0]:.3f}, {velocity_command[1]:.3f}, {velocity_command[2]:.3f}]")
                print(
                    f"Step {self.step_count}: Joy Cmd = [{joystick_command[0]:.3f}, {joystick_command[1]:.3f}, {joystick_command[2]:.3f}, {joystick_command[3]:.3f}]")

            # 自动保存检查点
            if self.step_count % self.save_interval == 0 and self.step_count > 0:
                self.auto_save_checkpoint()

            return True

        except Exception as e:
            print(f"数据采集失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_collection_status(self):
        """打印数据采集状态"""
        buffer_size = len(self.data_buffer['images'])
        elapsed_time = time.time() - self.start_time
        data_rate = self.step_count / elapsed_time if elapsed_time > 0 else 0

        print(f"\n=== V4数据采集状态 ===")
        print(f"总步数: {self.step_count}")
        print(f"缓冲区大小: {buffer_size}/{self.max_buffer_size}")
        print(f"Episode: {self.episode_count}")
        print(f"运行时间: {elapsed_time:.1f} 秒")
        print(f"数据率: {data_rate:.1f} 步/秒")

        if buffer_size > 0:
            latest_state = self.data_buffer['states'][-1]
            latest_vel_cmd = self.data_buffer['velocity_commands_world'][-1]
            latest_joy_cmd = self.data_buffer['joystick_commands_local'][-1]

            print(f"无人机位置: [{latest_state[0]:.2f}, {latest_state[1]:.2f}, {latest_state[2]:.2f}]")
            print(f"目标位置: [{self.drone.user_cmd.x:.2f}, {self.drone.user_cmd.y:.2f}, {self.drone.user_cmd.z:.2f}]")
            print(f"速度指令: [{latest_vel_cmd[0]:.3f}, {latest_vel_cmd[1]:.3f}, {latest_vel_cmd[2]:.3f}] m/s")
            print(
                f"手柄命令: [{latest_joy_cmd[0]:.3f}, {latest_joy_cmd[1]:.3f}, {latest_joy_cmd[2]:.3f}, {latest_joy_cmd[3]:.3f}]")

            # 显示任务进度
            if self.drone.finish_zone_pos is not None:
                distance = np.linalg.norm(latest_state[:3] - self.drone.finish_zone_pos)
                print(f"距离终点: {distance:.2f}m")

    def start_new_episode(self):
        """开始新的episode"""
        self.episode_count += 1
        self.start_time = time.time()
        self.data_saved = False
        self.clear_buffer()
        print(f"\n🎬 开始新的 Episode #{self.episode_count}")

    def auto_save_checkpoint(self):
        """自动保存检查点"""
        if self.step_count > 0 and not self.data_saved:
            filename = f"checkpoint_{self.episode_count:03d}_{self.step_count:08d}.h5"
            self.save_data(filename)
            print(f"自动保存检查点: {filename}")

    def clear_buffer(self):
        """清空数据缓冲区"""
        for key in self.data_buffer:
            if key != 'terrain_params':  # 保留地形参数
                self.data_buffer[key].clear()
        self.step_count = 0
        print("数据缓冲区已清空")

    def save_data(self, filename=None, episode_data=False):
        """保存采集的数据到HDF5文件"""
        if len(self.data_buffer['images']) == 0:
            print("没有数据可保存")
            return None

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_tag = f"episode_{self.episode_count:03d}" if episode_data else ""
            filename = f"expert_data_v4_{episode_tag}_{timestamp}.h5"

        filepath = os.path.join(self.data_dir, filename)

        try:
            with h5py.File(filepath, 'w') as f:
                # 保存主要数据
                images_array = np.array(self.data_buffer['images'])
                f.create_dataset('images', data=images_array, compression='gzip', compression_opts=9)
                f.create_dataset('states', data=np.array(self.data_buffer['states']), compression='gzip')
                f.create_dataset('velocity_commands_world',
                                 data=np.array(self.data_buffer['velocity_commands_world']),
                                 compression='gzip')
                f.create_dataset('joystick_commands_local',
                                 data=np.array(self.data_buffer['joystick_commands_local']),
                                 compression='gzip')
                f.create_dataset('timestamps', data=np.array(self.data_buffer['timestamps']))
                f.create_dataset('raw_joystick_inputs',
                                 data=np.array(self.data_buffer['raw_joystick_inputs']))

                # 保存详细数据
                f.create_dataset('positions', data=np.array(self.data_buffer['positions']))
                f.create_dataset('orientations', data=np.array(self.data_buffer['orientations']))
                f.create_dataset('velocities', data=np.array(self.data_buffer['velocities']))
                f.create_dataset('user_commands', data=np.array(self.data_buffer['user_commands']))
                f.create_dataset('motor_commands', data=np.array(self.data_buffer['motor_commands']))
                f.create_dataset('mission_status', data=np.array(self.data_buffer['mission_status']))
                f.create_dataset('target_positions', data=np.array(self.data_buffer['target_positions']))
                f.create_dataset('heading_angles', data=np.array(self.data_buffer['heading_angles']))

                # 保存地形参数
                if len(self.data_buffer['terrain_params']) > 0:
                    terrain_group = f.create_group('terrain_params')
                    for key, value in self.data_buffer['terrain_params'][0].items():
                        if isinstance(value, (int, float, str)):
                            terrain_group.attrs[key] = value
                        elif isinstance(value, np.ndarray):
                            terrain_group.create_dataset(key, data=value)

                # 保存元数据
                f.attrs['total_steps'] = self.step_count
                f.attrs['episode_count'] = self.episode_count
                f.attrs['image_shape'] = images_array[0].shape
                f.attrs['state_dim'] = len(self.data_buffer['states'][0])
                f.attrs['velocity_command_dim'] = 3
                f.attrs['joystick_command_dim'] = 4
                f.attrs['collection_date'] = datetime.datetime.now().isoformat()
                f.attrs['drone_mass'] = self.drone.se3_controller.m if hasattr(self.drone, 'se3_controller') else 1.325
                f.attrs['simulation_timestep'] = self.drone.m.opt.timestep
                f.attrs['mission_complete'] = self.drone.mission_complete

                if self.drone.finish_zone_pos is not None:
                    f.attrs['finish_zone_pos'] = self.drone.finish_zone_pos
                else:
                    f.attrs['finish_zone_pos'] = [0, 0, 0]

                # 保存控制参数
                f.attrs['xy_vel_scale'] = self.drone.XY_VEL_SCALE
                f.attrs['z_vel_scale'] = self.drone.Z_VEL_SCALE
                f.attrs['yaw_rate_scale'] = self.drone.YAW_RATE_SCALE
                f.attrs['controller_version'] = 'v4_compatible_with_low_freq_control_v2'

                # 保存无人机初始信息
                if hasattr(self.drone, 'initial_position'):
                    f.attrs['initial_position'] = self.drone.initial_position
                if hasattr(self.drone, 'initial_yaw'):
                    f.attrs['initial_yaw'] = self.drone.initial_yaw

            print(f"✅ 数据已保存: {filepath}")
            print(f"📊 包含 {len(self.data_buffer['images'])} 个数据点")

            # 打印数据统计
            if len(self.data_buffer['velocity_commands_world']) > 0:
                velocity_commands = np.array(self.data_buffer['velocity_commands_world'])
                print(f"📈 速度指令统计:")
                print(f"  Vx: [{velocity_commands[:, 0].min():.3f}, {velocity_commands[:, 0].max():.3f}]")
                print(f"  Vy: [{velocity_commands[:, 1].min():.3f}, {velocity_commands[:, 1].max():.3f}]")
                print(f"  Vz: [{velocity_commands[:, 2].min():.3f}, {velocity_commands[:, 2].max():.3f}]")

            if len(self.data_buffer['joystick_commands_local']) > 0:
                joystick_commands = np.array(self.data_buffer['joystick_commands_local'])
                print(f"🎮 手柄命令统计:")
                print(f"  Vx_local: [{joystick_commands[:, 0].min():.3f}, {joystick_commands[:, 0].max():.3f}]")
                print(f"  Vy_local: [{joystick_commands[:, 1].min():.3f}, {joystick_commands[:, 1].max():.3f}]")
                print(f"  Vz: [{joystick_commands[:, 2].min():.3f}, {joystick_commands[:, 2].max():.3f}]")
                print(f"  Yaw_rate: [{joystick_commands[:, 3].min():.3f}, {joystick_commands[:, 3].max():.3f}]")

            self.data_saved = True
            return filepath

        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def close(self):
        """关闭数据采集器"""
        if not self.data_saved and len(self.data_buffer['images']) > 0:
            print("正在保存最终数据...")
            self.save_data("final_collection_v4.h5")
        print("数据采集器已关闭")


def main_with_data_collection():
    """主函数 - 使用手柄控制采集数据，兼容low_freq_control_v2"""
    print("=" * 60)
    print("V4数据集采集器 - 兼容low_freq_control_v2")
    print("=" * 60)

    # 初始化无人机
    drone = Drone()

    # 初始化数据采集器
    data_collector = DataCollectorV4(drone, data_dir="v4_expert_data_trench_low_freq")

    # 设置摄像头
    if not data_collector.setup_camera():
        print("警告: 摄像头设置失败，将无法采集图像数据")
        return

    # 设置地形和初始位置
    drone.setup_sine_trench_terrain()

    # 确保终点位置在安全高度
    if drone.finish_zone_pos is not None:
        drone.finish_zone_pos[2] = max(drone.finish_zone_pos[2], 1.5)

    # 设置初始位置 - 使用沟壑起点
    trench_start_x = drone.trench_params['trench_end_x']
    trench_end_x = drone.trench_params['trench_start_x']
    amplitude = drone.trench_params['amplitude']
    wavelength = drone.trench_params['wavelength']

    # 选择起始点选项
    start_option = "entrance"  # 可选: "entrance", "center", "random_in_trench"

    if start_option == "entrance":
        start_x = trench_start_x
    elif start_option == "center":
        start_x = (trench_start_x + trench_end_x) / 2
    elif start_option == "random_in_trench":
        start_x = np.random.uniform(trench_start_x, trench_end_x)
    else:
        start_x = trench_start_x

    # 计算沟壑中心线在起始点的y坐标
    start_center_y = amplitude * np.sin(2 * np.pi * start_x / wavelength)
    start_height = 1.5  # 起始高度

    # 设置无人机初始位置
    drone.d.qpos[:3] = [start_x, start_center_y, start_height]

    # 计算初始偏航角
    initial_yaw = drone.calculate_initial_yaw()

    # 将偏航角转换为四元数
    half_yaw = initial_yaw * 0.5
    quaternion = [
        np.cos(half_yaw),  # w
        0.0,  # x
        0.0,  # y
        np.sin(half_yaw)  # z
    ]
    drone.d.qpos[3:7] = quaternion

    # 设置用户命令的偏航角和目标位置
    drone.user_cmd.yaw = initial_yaw
    drone.user_cmd.x = start_x
    drone.user_cmd.y = start_center_y
    drone.user_cmd.z = start_height

    # 保存初始位置和偏航角供数据记录使用
    drone.initial_position = [start_x, start_center_y, start_height]
    drone.initial_yaw = initial_yaw

    print(f"起始位置: x={start_x:.2f}, y={start_center_y:.2f}, z={start_height:.2f}")
    print(f"沟壑起点: {trench_start_x:.2f}, 沟壑终点: {trench_end_x:.2f}")
    print(f"无人机初始偏航角: {np.degrees(initial_yaw):.1f}°")

    # 设置终点位置
    if drone.finish_zone_pos is not None:
        drone.finish_zone_pos[2] = max(drone.finish_zone_pos[2], 1.5)
        finish_pos = drone.finish_zone_pos
        print(f"🎯 终点位置: [{finish_pos[0]:.2f}, {finish_pos[1]:.2f}, {finish_pos[2]:.2f}]")
        print(f"🎯 任务目标: 飞向终点区域（半径 {drone.FINISH_ZONE_RADIUS}m）")

    # 开始新的episode
    data_collector.start_new_episode()

    print("\n🎮 使用Xbox控制器控制无人机:")
    print("  左摇杆: 控制相对于无人机航向的水平(X, Y)速度")
    print("  右摇杆: 控制垂直(Z)速度和偏航率")
    print("  任务: 沿着沟壑飞行并到达终点区域")
    print("=" * 60)

    # 启动仿真查看器
    with mujoco.viewer.launch_passive(drone.m, drone.d) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

        # 设置初始渲染相机
        drone_eye_id = mujoco.mj_name2id(drone.m, mujoco.mjtObj.mjOBJ_CAMERA, "drone_eye")
        if drone_eye_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = drone_eye_id

        last_print_time = time.time()
        last_save_time = time.time()

        try:
            while viewer.is_running():
                step_start = time.time()

                # 如果任务完成，保存数据并退出
                if drone.mission_complete:
                    print("\n🎉 任务已完成！保存数据...")
                    # data_collector.save_data(f"mission_complete_episode_{data_collector.episode_count:03d}.h5")
                    data_collector.save_data(f"mission_complete_episode_50.h5")

                    # 等待2秒后退出
                    complete_time = time.time()
                    while time.time() - complete_time < 2.0:
                        mujoco.mj_step(drone.m, drone.d)
                        viewer.sync()
                        time.sleep(0.01)
                    break

                # 运行无人机控制（包括手柄输入处理）
                drone()

                # 采集数据
                data_collector.collect_step_data()

                # 定期状态打印（每2秒）
                if time.time() - last_print_time > 2.0:
                    current_vel_cmd = data_collector.current_velocity_command
                    current_joystick_cmd = data_collector.current_joystick_command
                    drone_pos = drone.state_estimator.base_pos

                    status_msg = (
                        f"Time: {drone.d.time:.2f}s | "
                        f"Steps: {data_collector.step_count} | "
                        f"Buffer: {len(data_collector.data_buffer['images'])} | "
                        f"Pos: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}] | "
                        f"Vel Cmd: [{current_vel_cmd[0]:.3f}, {current_vel_cmd[1]:.3f}, {current_vel_cmd[2]:.3f}] | "
                        f"Joy Cmd: [{current_joystick_cmd[0]:.3f}, {current_joystick_cmd[1]:.3f}]"
                    )
                    print(status_msg)
                    last_print_time = time.time()

                # 定期自动保存（每60秒）
                if time.time() - last_save_time > 60.0:
                    if data_collector.step_count > 0:
                        data_collector.auto_save_checkpoint()
                    last_save_time = time.time()

                # 步进仿真
                mujoco.mj_step(drone.m, drone.d)
                viewer.sync()

                # 实时性控制
                time_until_next_step = drone.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        except KeyboardInterrupt:
            print("\n⏹️ 用户中断，保存数据...")
            if not data_collector.data_saved and data_collector.step_count > 0:
                data_collector.save_data(f"interrupted_episode_{data_collector.episode_count:03d}.h5")

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()

            # 尝试保存数据
            if not data_collector.data_saved and data_collector.step_count > 0:
                data_collector.save_data(f"error_episode_{data_collector.episode_count:03d}.h5")

        finally:
            data_collector.close()
            print("仿真结束")


if __name__ == "__main__":
    main_with_data_collection()