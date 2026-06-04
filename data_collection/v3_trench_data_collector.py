# trench_data_colator_fixed.py
import os
import h5py
import datetime
import numpy as np
from collections import deque
import cv2
import time
import mujoco
from low_freq_control import Drone


class DataCollector:
    def __init__(self, drone_instance, data_dir="expert_data", max_buffer_size=50000):
        self.drone = drone_instance
        self.data_dir = data_dir
        self.max_buffer_size = max_buffer_size

        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)

        # 数据缓冲区 - 添加 joystick_commands_local
        self.data_buffer = {
            'images': deque(maxlen=max_buffer_size),
            'states': deque(maxlen=max_buffer_size),
            'velocity_commands_world': deque(maxlen=max_buffer_size),
            'joystick_commands_local': deque(maxlen=max_buffer_size),  # 新增：四维手柄命令
            'timestamps': deque(maxlen=max_buffer_size),
            'positions': deque(maxlen=max_buffer_size),
            'orientations': deque(maxlen=max_buffer_size),
            'velocities': deque(maxlen=max_buffer_size),
            'user_commands': deque(maxlen=max_buffer_size),
            'motor_commands': deque(maxlen=max_buffer_size),
            'mission_status': deque(maxlen=max_buffer_size),
            'raw_joystick_inputs': deque(maxlen=max_buffer_size)
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
        self.current_joystick_command = np.zeros(4)  # 新增：当前四维手柄命令
        self.current_raw_inputs = np.zeros(4)

        print(f"数据采集器初始化完成，数据将保存在: {data_dir}")

    def setup_camera(self):
        """设置无人机摄像头用于图像采集"""
        try:
            camera_id = mujoco.mj_name2id(self.drone.m, mujoco.mjtObj.mjOBJ_CAMERA, "drone_eye")
            if camera_id >= 0:
                self.camera_id = camera_id
                print(f"找到无人机摄像头: drone_eye (ID: {camera_id})")
            else:
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

            return image

        except Exception as e:
            print(f"图像捕获失败: {e}")
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

    def get_drone_state(self):
        """获取无人机完整状态信息"""
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

    def get_velocity_command(self):
        """获取当前的速度指令 - 直接从Drone类中获取"""
        try:
            # 方法1：尝试直接访问Drone类中的速度指令属性
            if hasattr(self.drone, 'current_velocity_command'):
                velocity_command = self.drone.current_velocity_command
                if velocity_command is not None:
                    return velocity_command

            # 方法2：尝试访问手柄处理后的速度指令
            if hasattr(self.drone, 'v_x_world') and hasattr(self.drone, 'v_y_world') and hasattr(self.drone, 'v_z'):
                return np.array([self.drone.v_x_world, self.drone.v_y_world, self.drone.v_z])

            # 方法3：基于目标位置计算期望速度（作为备选）
            target_pos = np.array([self.drone.user_cmd.x, self.drone.user_cmd.y, self.drone.user_cmd.z])
            current_pos = self.drone.state_estimator.base_pos

            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance > 0.1:
                direction_normalized = direction / distance
                desired_speed = min(1.0, distance * 2.0)
                velocity_command = direction_normalized * desired_speed
            else:
                velocity_command = np.array([0.0, 0.0, 0.0])

            return velocity_command

        except Exception as e:
            print(f"获取速度指令失败: {e}")
            return np.array([0.0, 0.0, 0.0])

    def get_joystick_command_local(self):
        """获取四维手柄控制命令（局部坐标系）"""
        try:
            # 直接从Drone类中获取处理后的手柄命令
            joystick_cmd = np.array([0.0, 0.0, 0.0, 0.0])

            # 获取局部坐标系的速度命令
            if hasattr(self.drone, 'v_x_local_processed'):
                joystick_cmd[0] = self.drone.v_x_local_processed  # x方向局部速度
                joystick_cmd[1] = self.drone.v_y_local_processed  # y方向局部速度
                joystick_cmd[2] = self.drone.v_z_processed  # z方向速度
                joystick_cmd[3] = self.drone.yaw_rate_processed  # 期望偏航角速度
            else:
                # 备选方法：从手柄输入获取
                raw_inputs = self.get_raw_joystick_inputs()
                joystick_cmd[0] = raw_inputs[1] * self.drone.XY_VEL_SCALE  # v_x_local
                joystick_cmd[1] = raw_inputs[0] * self.drone.XY_VEL_SCALE  # v_y_local
                joystick_cmd[2] = raw_inputs[2] * self.drone.Z_VEL_SCALE  # v_z
                joystick_cmd[3] = raw_inputs[3] * self.drone.YAW_RATE_SCALE  # yaw_rate

            return joystick_cmd

        except Exception as e:
            print(f"获取手柄命令失败: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0])

    def get_raw_joystick_inputs(self):
        """获取原始手柄输入用于调试"""
        try:
            # 尝试获取原始手柄输入
            raw_inputs = np.array([0.0, 0.0, 0.0, 0.0])

            if hasattr(self.drone, 'joystick'):
                joystick = self.drone.joystick
                if joystick is not None:
                    # 获取各个轴的原始值
                    raw_inputs[0] = joystick.get_axis(0) if joystick.get_numbuttons() > 0 else 0.0  # v_y_local
                    raw_inputs[1] = joystick.get_axis(1) if joystick.get_numbuttons() > 1 else 0.0  # v_x_local
                    raw_inputs[2] = -joystick.get_axis(3) if joystick.get_numbuttons() > 3 else 0.0  # v_z
                    raw_inputs[3] = -joystick.get_axis(2) if joystick.get_numbuttons() > 2 else 0.0  # yaw_rate

            return raw_inputs

        except Exception as e:
            print(f"获取原始手柄输入失败: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0])

    def collect_step_data(self):
        """收集单步数据"""
        try:
            # 捕获图像
            image = self.capture_image()

            # 获取状态信息
            state = self.get_drone_state()

            # 获取当前世界坐标系速度指令
            velocity_command = self.get_velocity_command()

            # 获取四维手柄控制命令（局部坐标系）- 新增
            joystick_command = self.get_joystick_command_local()

            # 获取原始手柄输入（用于调试）
            raw_inputs = self.get_raw_joystick_inputs()

            # 获取当前电机指令
            motor_command = self.drone.d.ctrl[:4].copy()

            # 获取用户命令
            user_cmd = np.array([self.drone.user_cmd.x, self.drone.user_cmd.y,
                                 self.drone.user_cmd.z, self.drone.user_cmd.yaw])

            # 获取时间戳
            timestamp = self.drone.d.time

            # 获取任务状态
            mission_status = 1 if self.drone.mission_complete else 0

            # 存储数据 - 添加joystick_commands_local
            self.data_buffer['images'].append(image)
            self.data_buffer['states'].append(state)
            self.data_buffer['velocity_commands_world'].append(velocity_command)
            self.data_buffer['joystick_commands_local'].append(joystick_command)  # 新增
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

            # 更新当前指令
            self.current_velocity_command = velocity_command
            self.current_joystick_command = joystick_command  # 新增
            self.current_raw_inputs = raw_inputs

            self.step_count += 1

            # 定期显示采集状态和调试信息
            if self.step_count % 500 == 0:
                self.print_collection_status()

            # 每100步打印一次调试信息
            if self.step_count % 100 == 0:
                print(
                    f"Step {self.step_count}: Velocity Command = [{velocity_command[0]:.3f}, {velocity_command[1]:.3f}, {velocity_command[2]:.3f}]")
                print(
                    f"Step {self.step_count}: Joystick Command = [{joystick_command[0]:.3f}, {joystick_command[1]:.3f}, {joystick_command[2]:.3f}, {joystick_command[3]:.3f}]")
                print(
                    f"Step {self.step_count}: Raw Inputs = [{raw_inputs[0]:.3f}, {raw_inputs[1]:.3f}, {raw_inputs[2]:.3f}, {raw_inputs[3]:.3f}]")

            # 自动保存检查点
            if self.step_count % self.save_interval == 0:
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

        print(f"\n=== 数据采集状态 ===")
        print(f"总步数: {self.step_count}")
        print(f"缓冲区大小: {buffer_size}/{self.max_buffer_size}")
        print(f"Episode: {self.episode_count}")
        print(f"运行时间: {elapsed_time:.1f} 秒")
        print(f"数据率: {data_rate:.1f} 步/秒")

        if buffer_size > 0:
            latest_state = self.data_buffer['states'][-1]
            latest_vel_cmd = self.data_buffer['velocity_commands_world'][-1]
            latest_joystick_cmd = self.data_buffer['joystick_commands_local'][-1]  # 新增
            latest_raw = self.data_buffer['raw_joystick_inputs'][-1]
            print(f"位置: [{latest_state[0]:.2f}, {latest_state[1]:.2f}, {latest_state[2]:.2f}]")
            print(f"速度指令: [{latest_vel_cmd[0]:.3f}, {latest_vel_cmd[1]:.3f}, {latest_vel_cmd[2]:.3f}] m/s")
            print(
                f"手柄命令: [{latest_joystick_cmd[0]:.3f}, {latest_joystick_cmd[1]:.3f}, {latest_joystick_cmd[2]:.3f}, {latest_joystick_cmd[3]:.3f}]")  # 新增
            print(f"原始输入: [{latest_raw[0]:.3f}, {latest_raw[1]:.3f}, {latest_raw[2]:.3f}, {latest_raw[3]:.3f}]")

    def start_new_episode(self):
        """开始新的episode"""
        self.episode_count += 1
        self.start_time = time.time()
        self.data_saved = False
        print(f"\n🎬 开始新的 Episode #{self.episode_count}")

    def auto_save_checkpoint(self):
        """自动保存检查点"""
        if self.step_count > 0 and not self.data_saved:
            filename = f"checkpoint_{self.step_count:08d}.h5"
            self.save_data(filename)

    def clear_buffer(self):
        """清空数据缓冲区"""
        for key in self.data_buffer:
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
            filename = f"expert_data_{episode_tag}_{timestamp}.h5"

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
                f.create_dataset('joystick_commands_local',  # 新增
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

                # 保存元数据
                f.attrs['total_steps'] = self.step_count
                f.attrs['episode_count'] = self.episode_count
                f.attrs['image_shape'] = images_array[0].shape
                f.attrs['state_dim'] = len(self.data_buffer['states'][0])
                f.attrs['velocity_command_dim'] = 3
                f.attrs['joystick_command_dim'] = 4  # 新增
                f.attrs['collection_date'] = datetime.datetime.now().isoformat()
                f.attrs['drone_mass'] = self.drone.se3_controller.m
                f.attrs['simulation_timestep'] = self.drone.m.opt.timestep
                f.attrs['mission_complete'] = self.drone.mission_complete
                f.attrs['finish_zone_pos'] = self.drone.finish_zone_pos if self.drone.finish_zone_pos is not None else [
                    0, 0, 0]

                # 保存控制参数
                f.attrs['xy_vel_scale'] = self.drone.XY_VEL_SCALE
                f.attrs['z_vel_scale'] = self.drone.Z_VEL_SCALE
                f.attrs['yaw_rate_scale'] = self.drone.YAW_RATE_SCALE

                # 保存地形参数
                if hasattr(self.drone, 'trench_params'):
                    for key, value in self.drone.trench_params.items():
                        f.attrs[f'trench_{key}'] = value

            print(f"数据已保存: {filepath}")
            print(f"包含 {len(self.data_buffer['images'])} 个数据点")

            # 打印速度指令统计
            velocity_commands = np.array(self.data_buffer['velocity_commands_world'])
            print(f"速度指令统计:")
            print(f"  Vx: [{velocity_commands[:, 0].min():.3f}, {velocity_commands[:, 0].max():.3f}]")
            print(f"  Vy: [{velocity_commands[:, 1].min():.3f}, {velocity_commands[:, 1].max():.3f}]")
            print(f"  Vz: [{velocity_commands[:, 2].min():.3f}, {velocity_commands[:, 2].max():.3f}]")

            # 打印手柄命令统计 - 新增
            joystick_commands = np.array(self.data_buffer['joystick_commands_local'])
            print(f"手柄命令统计:")
            print(f"  Vx_local: [{joystick_commands[:, 0].min():.3f}, {joystick_commands[:, 0].max():.3f}]")
            print(f"  Vy_local: [{joystick_commands[:, 1].min():.3f}, {joystick_commands[:, 1].max():.3f}]")
            print(f"  Vz: [{joystick_commands[:, 2].min():.3f}, {joystick_commands[:, 2].max():.3f}]")
            print(f"  Yaw_rate: [{joystick_commands[:, 3].min():.3f}, {joystick_commands[:, 3].max():.3f}]")

            # 打印原始输入统计
            raw_inputs = np.array(self.data_buffer['raw_joystick_inputs'])
            print(f"原始输入统计:")
            print(f"  Axis 0: [{raw_inputs[:, 0].min():.3f}, {raw_inputs[:, 0].max():.3f}]")
            print(f"  Axis 1: [{raw_inputs[:, 1].min():.3f}, {raw_inputs[:, 1].max():.3f}]")
            print(f"  Axis 2: [{raw_inputs[:, 2].min():.3f}, {raw_inputs[:, 2].max():.3f}]")
            print(f"  Axis 3: [{raw_inputs[:, 3].min():.3f}, {raw_inputs[:, 3].max():.3f}]")

            self.data_saved = True
            return filepath

        except Exception as e:
            print(f"保存数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def close(self):
        """关闭数据采集器"""
        if not self.data_saved and len(self.data_buffer['images']) > 0:
            self.save_data("final_collection.h5")
        print("数据采集器已关闭")


# 同时需要修改low_freq_control.py中的Drone类，添加处理后的手柄命令属性
"""
在Drone类的update_user_command_from_joystick方法中添加：

class Drone:
    def update_user_command_from_joystick(self):
        # ... 现有代码 ...

        # 在应用死区后，存储处理后的手柄命令
        self.v_x_local_processed = v_x_local
        self.v_y_local_processed = v_y_local  
        self.v_z_processed = v_z
        self.yaw_rate_processed = yaw_rate

        # ... 其余代码 ...
"""


def main_with_data_collection():
    """主函数 - 使用手柄控制采集数据"""
    # 初始化无人机
    drone = Drone()

    # 初始化数据采集器
    data_collector = DataCollector(drone, data_dir="v3_expert_data_trench_low_freq")

    # 设置摄像头
    if not data_collector.setup_camera():
        print("警告: 摄像头设置失败，将无法采集图像数据")
        return

    # 设置地形和初始位置
    drone.setup_sine_trench_terrain()

    # 确保终点位置在安全高度
    if drone.finish_zone_pos is not None:
        drone.finish_zone_pos[2] = max(drone.finish_zone_pos[2], 1.5)

    # 设置初始位置
    trench_start_x = drone.trench_params['trench_end_x']
    amplitude = drone.trench_params['amplitude']
    wavelength = drone.trench_params['wavelength']
    entrance_center_y = amplitude * np.sin(2 * np.pi * trench_start_x / wavelength)
    entrance_height = 1.5

    drone.d.qpos[:3] = [7.0, entrance_center_y, entrance_height]
    drone.d.qpos[3:7] = [1, 0, 0, 0]

    # 设置目标位置
    drone.user_cmd.x = 7.0
    drone.user_cmd.y = entrance_center_y
    drone.user_cmd.z = entrance_height

    print(f"沟壑入口位置: x={7.0:.2f}, y={entrance_center_y:.2f}")
    print(f"使用手柄控制无人机飞行，采集四维手柄控制命令数据")
    print(f"调试信息将每100步显示一次")

    # 开始新的episode
    data_collector.start_new_episode()

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

        try:
            while viewer.is_running():
                step_start = time.time()

                # 如果任务完成，退出
                if drone.mission_complete:
                    print("🎉 任务已完成！保存数据...")
                    data_collector.save_data(f"mission_complete_episode_58.h5")
                    break

                # 运行无人机控制（包括手柄输入处理）
                drone()

                # 采集数据
                data_collector.collect_step_data()

                # 定期状态打印
                if time.time() - last_print_time > 2.0:
                    current_vel_cmd = data_collector.current_velocity_command
                    current_joystick_cmd = data_collector.current_joystick_command
                    status_msg = (
                        f"Time: {drone.d.time:.2f}s | "
                        f"Steps: {data_collector.step_count} | "
                        f"Buffer: {len(data_collector.data_buffer['images'])} | "
                        f"速度指令: [{current_vel_cmd[0]:.3f}, {current_vel_cmd[1]:.3f}, {current_vel_cmd[2]:.3f}] m/s | "
                        f"手柄命令: [{current_joystick_cmd[0]:.3f}, {current_joystick_cmd[1]:.3f}, {current_joystick_cmd[2]:.3f}, {current_joystick_cmd[3]:.3f}]"
                    )
                    print(status_msg)
                    last_print_time = time.time()

                # 步进仿真
                mujoco.mj_step(drone.m, drone.d)
                viewer.sync()

                # 实时性控制
                time_until_next_step = drone.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        except KeyboardInterrupt:
            print("\n用户中断，保存数据...")
            if not data_collector.data_saved:
                data_collector.save_data("interrupted_collection.h5")
        finally:
            data_collector.close()


if __name__ == "__main__":
    main_with_data_collection()