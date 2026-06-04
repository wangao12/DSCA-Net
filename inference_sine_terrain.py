# inference_DualStreamLSTM_improved.py
# 适配改进版训练代码（幅角+置信度+跨模态注意力）的推理测试脚本

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mujoco
import time
import h5py
from collections import deque
import matplotlib.pyplot as plt
from low_freq_control import Drone, UserCommand  # 请确保此模块可用


# ======================== 光流后处理：幅度-方向 + 置信度 ========================
def flow_to_magnitude_angle_confidence(flow_uv, conf_threshold_quantile=0.3):
    """
    将 (u,v) 光流转换为：
        magnitude : 归一化幅度
        angle_sin, angle_cos : 方向的正余弦
        confidence : 基于幅度的置信度掩码
    返回 mag_ang: (H, W, 3), conf: (H, W, 1)
    """
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    mag = np.sqrt(u**2 + v**2) + 1e-8

    # 方向 sin/cos
    sin_angle = v / mag
    cos_angle = u / mag

    # 置信度：基于幅度的指数函数
    mag_thresh = np.quantile(mag, conf_threshold_quantile)
    confidence = 1.0 - np.exp(-mag / (mag_thresh + 1e-8))
    confidence = np.clip(confidence, 0.1, 1.0)

    mag_ang = np.stack([mag, sin_angle, cos_angle], axis=-1)
    confidence = confidence[..., np.newaxis]
    return mag_ang, confidence


# ======================== 跨模态注意力模块（与训练代码一致）========================
class CrossModalAttention(nn.Module):
    """交叉模态注意力：光流引导RGB，RGB修正光流"""
    def __init__(self, rgb_channels, flow_channels, reduction=8):
        super().__init__()
        self.rgb_query = nn.Conv2d(rgb_channels, rgb_channels // reduction, 1)
        self.flow_key = nn.Conv2d(flow_channels, rgb_channels // reduction, 1)
        self.flow_value = nn.Conv2d(flow_channels, rgb_channels, 1)

        self.flow_query = nn.Conv2d(flow_channels, flow_channels // reduction, 1)
        self.rgb_key = nn.Conv2d(rgb_channels, flow_channels // reduction, 1)
        self.rgb_value = nn.Conv2d(rgb_channels, flow_channels, 1)

        self.rgb_out = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 1),
            nn.BatchNorm2d(rgb_channels)
        )
        self.flow_out = nn.Sequential(
            nn.Conv2d(flow_channels, flow_channels, 1),
            nn.BatchNorm2d(flow_channels)
        )

    def forward(self, rgb_feat, flow_feat):
        B, _, H, W = rgb_feat.shape
        Q_r = self.rgb_query(rgb_feat).view(B, -1, H*W)
        K_f = self.flow_key(flow_feat).view(B, -1, H*W)
        V_f = self.flow_value(flow_feat).view(B, -1, H*W)
        attn_r = F.softmax(torch.bmm(Q_r.transpose(1,2), K_f), dim=-1)
        rgb_attended = torch.bmm(V_f, attn_r.transpose(1,2)).view(B, -1, H, W)
        rgb_enhanced = self.rgb_out(rgb_feat + rgb_attended)

        Q_f = self.flow_query(flow_feat).view(B, -1, H*W)
        K_r = self.rgb_key(rgb_feat).view(B, -1, H*W)
        V_r = self.rgb_value(rgb_feat).view(B, -1, H*W)
        attn_f = F.softmax(torch.bmm(Q_f.transpose(1,2), K_r), dim=-1)
        flow_attended = torch.bmm(V_r, attn_f.transpose(1,2)).view(B, -1, H, W)
        flow_enhanced = self.flow_out(flow_feat + flow_attended)

        return rgb_enhanced, flow_enhanced


# ======================== 增强视觉编码器（支持返回中间特征图）========================
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class EnhancedVisionEncoder(nn.Module):
    def __init__(self, image_size=(240, 320), in_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.block1 = CNNBlock(32, 64)
        self.block2 = CNNBlock(64, 128)
        self.block3 = CNNBlock(128, 256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 4))

        self.feature_compress = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )

    def forward(self, x, return_intermediate=False):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)  # (B, 256, 3, 4)
        intermediate = x
        x = self.feature_compress(x)
        if return_intermediate:
            return x, intermediate
        return x


# ======================== 双流 LSTM 模型（改进版）========================
class DualStreamLSTMDroneModel(nn.Module):
    def __init__(self, sequence_length=5, joystick_command_dim=4,
                 image_size=(240, 320), fusion_method='concat'):
        super().__init__()
        self.sequence_length = sequence_length
        self.fusion_method = fusion_method

        self.rgb_encoder = EnhancedVisionEncoder(image_size=image_size, in_channels=3)
        self.flow_encoder = EnhancedVisionEncoder(image_size=image_size, in_channels=3)  # 3通道：mag, sin, cos

        self.cross_attn = CrossModalAttention(rgb_channels=256, flow_channels=256, reduction=8)
        self.conf_downsample = nn.AdaptiveAvgPool2d((3, 4))

        self.rgb_lstm = nn.LSTM(2048, 128, batch_first=True, num_layers=2, dropout=0.3)
        self.flow_lstm = nn.LSTM(2048, 64, batch_first=True, num_layers=2, dropout=0.3)

        self.motion_enhancer = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh()
        )

        fusion_input_dim = 128 + 64
        if fusion_method == 'attention':
            self.channel_gate = nn.Sequential(
                nn.Linear(fusion_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, fusion_input_dim),
                nn.Sigmoid()
            )
        else:
            self.channel_gate = None

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.joystick_command_predictor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, joystick_command_dim)
        )

    def forward(self, rgb, flow_mag_ang, flow_conf):
        batch_size, seq_len = rgb.shape[0], rgb.shape[1]

        rgb_flat = rgb.reshape(batch_size * seq_len, *rgb.shape[2:])
        flow_flat = flow_mag_ang.reshape(batch_size * seq_len, *flow_mag_ang.shape[2:])
        conf_flat = flow_conf.reshape(batch_size * seq_len, *flow_conf.shape[2:])

        rgb_vec, rgb_inter = self.rgb_encoder(rgb_flat, return_intermediate=True)
        flow_vec, flow_inter = self.flow_encoder(flow_flat, return_intermediate=True)

        rgb_inter_enh, flow_inter_enh = self.cross_attn(rgb_inter, flow_inter)

        conf_down = self.conf_downsample(conf_flat)
        flow_inter_weighted = flow_inter_enh * conf_down

        rgb_feat_final = self.rgb_encoder.feature_compress(rgb_inter_enh)
        flow_feat_final = self.flow_encoder.feature_compress(flow_inter_weighted)

        rgb_seq = rgb_feat_final.reshape(batch_size, seq_len, -1)
        flow_seq = flow_feat_final.reshape(batch_size, seq_len, -1)

        rgb_lstm_out, _ = self.rgb_lstm(rgb_seq)
        flow_lstm_out, _ = self.flow_lstm(flow_seq)

        rgb_last = rgb_lstm_out[:, -1, :]
        flow_last = flow_lstm_out[:, -1, :]

        enhanced_flow = self.motion_enhancer(flow_last)
        combined = torch.cat([rgb_last, enhanced_flow], dim=1)

        if self.fusion_method == 'attention' and self.channel_gate is not None:
            gate = self.channel_gate(combined)
            combined = combined * gate

        fused = self.fusion(combined)
        return self.joystick_command_predictor(fused)


# ======================== 光流计算工具类（仅计算原始uv）========================
class OpticalFlowCalculator:
    def __init__(self, method='farneback'):
        self.method = method

    def calculate_flow(self, img1, img2):
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()

        if len(img1.shape) == 3 and img1.shape[0] == 3:
            img1 = img1.transpose(1, 2, 0)
            img2 = img2.transpose(1, 2, 0)

        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1
            gray2 = img2

        if gray1.dtype != np.uint8:
            gray1 = (gray1 * 255).astype(np.uint8)
            gray2 = (gray2 * 255).astype(np.uint8)

        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        elif self.method == 'lk':
            h, w = gray1.shape
            flow = np.zeros((h, w, 2), dtype=np.float32)
            p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)
            if p0 is not None:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                for (new, old) in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    flow[int(b), int(a)] = [a - c, b - d]
                flow[..., 0] = cv2.medianBlur(flow[..., 0], 5)
                flow[..., 1] = cv2.medianBlur(flow[..., 1], 5)
        else:
            raise ValueError(f"未知的光流方法: {self.method}")
        return flow


# ======================== 视频录制器 ========================
class VideoRecorder:
    def __init__(self, output_path, fps=30, frame_size=(320, 240)):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.video_writer = None

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        print(f"开始录制视频: {self.output_path}")

    def add_frame(self, frame):
        if self.video_writer is not None:
            if frame.shape[:2] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)

    def stop(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"视频录制完成: {self.output_path}")
            self.video_writer = None


# ======================== 推理控制器（适配改进版模型）========================
class DroneInferenceController:
    def __init__(self, model_path, normalization_path, sequence_length=5,
                 image_size=(240, 320), flow_method='farneback', fusion_method='attention'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.sequence_length = sequence_length
        self.image_size = image_size
        self.flow_method = flow_method
        self.fusion_method = fusion_method

        self.flow_calculator = OpticalFlowCalculator(method=flow_method)

        self.model = self.load_model(model_path)
        self.model.eval()

        self.normalization_params = np.load(normalization_path, allow_pickle=True).item()

        self.drone_controller = Drone()
        self.user_cmd = UserCommand()

        # 三个序列：RGB图像、幅角光流、置信度掩码
        self.image_sequence = deque(maxlen=sequence_length)
        self.flow_mag_ang_sequence = deque(maxlen=sequence_length)
        self.flow_conf_sequence = deque(maxlen=sequence_length)
        self.prev_image = None

        self.command_history = deque(maxlen=100)
        self.inference_times = []

        print(f"改进版双流 LSTM 推理控制器初始化完成 (幅角+置信度+跨模态注意力)")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        seq_len = config.get('sequence_length', 5)
        cmd_dim = config.get('joystick_command_dim', 4)
        img_size = config.get('image_size', (240, 320))
        fusion_method = config.get('fusion_method', 'attention')

        model = DualStreamLSTMDroneModel(
            sequence_length=seq_len,
            joystick_command_dim=cmd_dim,
            image_size=img_size,
            fusion_method=fusion_method
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        print(f"模型已加载: {model_path}")
        return model

    def preprocess_image(self, image):
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(image).float()

    def preprocess_flow(self, flow_uv):
        """将原始 uv 光流转换为幅角三通道和置信度单通道"""
        if flow_uv.shape[:2] != self.image_size:
            flow_uv = cv2.resize(flow_uv, (self.image_size[1], self.image_size[0]))
        # 生成幅角与置信度
        mag_ang, conf = flow_to_magnitude_angle_confidence(flow_uv)
        # 幅度归一化（除以图像尺寸）
        mag_ang[..., 0] /= max(self.image_size[1], self.image_size[0])
        # 转为 CHW
        mag_ang = mag_ang.transpose(2, 0, 1).astype(np.float32)
        conf = conf.transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(mag_ang).float(), torch.from_numpy(conf).float()

    def add_to_sequence(self, image):
        processed_image = self.preprocess_image(image)

        if self.prev_image is not None:
            flow_uv = self.flow_calculator.calculate_flow(self.prev_image, image)
            mag_ang_tensor, conf_tensor = self.preprocess_flow(flow_uv)
        else:
            # 第一帧用零填充
            mag_ang_tensor = torch.zeros((3, self.image_size[0], self.image_size[1]))
            conf_tensor = torch.zeros((1, self.image_size[0], self.image_size[1]))

        self.prev_image = image.copy()
        self.image_sequence.append(processed_image)
        self.flow_mag_ang_sequence.append(mag_ang_tensor)
        self.flow_conf_sequence.append(conf_tensor)

        # 序列长度不足时重复第一帧
        while len(self.image_sequence) < self.sequence_length:
            self.image_sequence.appendleft(processed_image)
            self.flow_mag_ang_sequence.appendleft(mag_ang_tensor)
            self.flow_conf_sequence.appendleft(conf_tensor)

    def get_sequence_tensors(self):
        if len(self.image_sequence) < self.sequence_length:
            raise ValueError(f"序列长度不足: {len(self.image_sequence)} < {self.sequence_length}")
        rgb_tensor = torch.stack(list(self.image_sequence)).unsqueeze(0).to(self.device)
        flow_mag_ang_tensor = torch.stack(list(self.flow_mag_ang_sequence)).unsqueeze(0).to(self.device)
        flow_conf_tensor = torch.stack(list(self.flow_conf_sequence)).unsqueeze(0).to(self.device)
        return rgb_tensor, flow_mag_ang_tensor, flow_conf_tensor

    def postprocess_command(self, joystick_command):
        mean = self.normalization_params['joystick_command_mean']
        std = self.normalization_params['joystick_command_std']
        return joystick_command * std + mean

    def predict_control(self, image):
        self.add_to_sequence(image)

        with torch.no_grad():
            rgb, flow_mag_ang, flow_conf = self.get_sequence_tensors()
            start_time = time.time()
            joystick_command_norm = self.model(rgb, flow_mag_ang, flow_conf)
            inference_time = time.time() - start_time

            self.inference_times.append(inference_time)
            cmd = joystick_command_norm.cpu().numpy()[0]
            cmd_denorm = self.postprocess_command(cmd)
            self.command_history.append(cmd_denorm.copy())
            return cmd_denorm

    def get_inference_stats(self):
        if not self.inference_times:
            return {'avg_time': 0, 'fps': 0}
        total = sum(self.inference_times)
        return {
            'avg_time': np.mean(self.inference_times) * 1000,
            'fps': len(self.inference_times) / total if total > 0 else 0,
            'num_inferences': len(self.inference_times)
        }

    def apply_joystick_commands(self, joystick_command):
        vx, vy, vz, yr = joystick_command
        self.drone_controller.v_x_local_processed = vx
        self.drone_controller.v_y_local_processed = vy
        self.drone_controller.v_z_processed = vz
        self.drone_controller.yaw_rate_processed = yr
        self._update_user_command_from_neural_network(vx, vy, vz, yr)

    def _update_user_command_from_neural_network(self, vx, vy, vz, yr):
        dt = self.drone_controller.m.opt.timestep
        self.user_cmd.yaw += yr * self.drone_controller.YAW_RATE_SCALE * dt
        c, s = np.cos(self.user_cmd.yaw), np.sin(self.user_cmd.yaw)
        vx_w = (vx * c - vy * s) * self.drone_controller.XY_VEL_SCALE
        vy_w = (vx * s + vy * c) * self.drone_controller.XY_VEL_SCALE
        self.user_cmd.x += vx_w * dt
        self.user_cmd.y += vy_w * dt
        self.user_cmd.z += vz * self.drone_controller.Z_VEL_SCALE * dt
        self.user_cmd.z = np.clip(self.user_cmd.z, 0.2, 2.5)

        drone_pos = self.drone_controller.state_estimator.pos
        target = np.array([self.user_cmd.x, self.user_cmd.y, self.user_cmd.z])
        error = target - drone_pos
        dist = np.linalg.norm(error)
        if dist > self.drone_controller.MAX_TARGET_DISTANCE:
            target = drone_pos + error / dist * self.drone_controller.MAX_TARGET_DISTANCE
            self.user_cmd.x, self.user_cmd.y, self.user_cmd.z = target

    def compute_control(self):
        self.drone_controller.user_cmd = self.user_cmd
        self.drone_controller()
        return self.drone_controller.d.ctrl[:4].copy()


# ======================== 状态估计器 ========================
class InferenceStateEstimator:
    def __init__(self, m, d):
        self.m = m
        self.d = d
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.pos = np.zeros(3)
        self.R = np.eye(3)

    def update(self):
        self.base_pos = self.d.qpos[:3].copy()
        self.base_quat = self.d.qpos[3:7].copy()
        self.pos = self.base_pos.copy()
        w, x, y, z = self.base_quat
        self.R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])


# ======================== 辅助函数 ========================
def add_text_overlay(image, text_lines):
    overlay = image.copy()
    h, w = overlay.shape[:2]
    bg_h = min(len(text_lines) * 30 + 10, h)
    bg = np.zeros((bg_h, w, 3), dtype=np.uint8)
    overlay[:bg_h] = cv2.addWeighted(overlay[:bg_h], 0.3, bg, 0.7, 0)
    for i, line in enumerate(text_lines):
        y = 25 + i * 30
        if y < h:
            cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return overlay


# ======================== 主推理函数 ========================
def main_inference_test(model_dir=None, episode_num=50, record_video=True,
                        flow_method='farneback', fusion_method='attention'):
    if model_dir is None:
        model_dir = f"DualStream_Improved_{episode_num}_ep_seq5_ds1_flow_{flow_method}_fusion_{fusion_method}"

    model_path = os.path.join(model_dir, "best_model.pth")
    normalization_path = os.path.join(model_dir, "normalization_params.npy")

    if not os.path.exists(model_path) or not os.path.exists(normalization_path):
        print(f"错误: 找不到模型文件或归一化文件")
        print(f"模型路径: {model_path}")
        print(f"归一化路径: {normalization_path}")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    sequence_length = config.get('sequence_length', 5)

    controller = DroneInferenceController(
        model_path, normalization_path,
        sequence_length=sequence_length,
        flow_method=flow_method,
        fusion_method=fusion_method
    )

    m = controller.drone_controller.m
    d = controller.drone_controller.d
    state_estimator = InferenceStateEstimator(m, d)
    controller.drone_controller.state_estimator = state_estimator

    controller.drone_controller.setup_sine_trench_terrain()
    if controller.drone_controller.finish_zone_pos is not None:
        controller.drone_controller.finish_zone_pos[2] = max(controller.drone_controller.finish_zone_pos[2], 1.5)

    trench_start_x = controller.drone_controller.trench_params['trench_end_x']
    amplitude = controller.drone_controller.trench_params['amplitude']
    wavelength = controller.drone_controller.trench_params['wavelength']
    entrance_center_y = amplitude * np.sin(2 * np.pi * trench_start_x / wavelength)
    entrance_height = 1.5

    d.qpos[:3] = [7, entrance_center_y, entrance_height]
    d.qpos[3:7] = [1, 0, 0, 0]
    controller.user_cmd.x = 7
    controller.user_cmd.y = entrance_center_y
    controller.user_cmd.z = entrance_height

    renderer = mujoco.Renderer(m, 240, 320)
    camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "drone_eye")
    if camera_id < 0:
        camera_id = 0

    video_recorder = None
    if record_video:
        os.makedirs("inference_videos", exist_ok=True)
        video_path = os.path.join("inference_videos", f"inference_improved_ep{episode_num}.mp4")
        video_recorder = VideoRecorder(video_path, fps=30, frame_size=(320, 240))
        video_recorder.start()

    print(f"改进版双流 LSTM 推理测试开始 (序列长度={sequence_length})")
    print(f"入口: x={trench_start_x:.2f}, y={entrance_center_y:.2f}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        drone_eye_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "drone_eye")
        if drone_eye_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = drone_eye_id

        step_count = 0
        mission_complete = False
        last_print_time = time.time()

        try:
            while viewer.is_running() and not mission_complete:
                if controller.drone_controller.mission_complete:
                    mission_complete = True
                    print("🎉 任务已完成！")
                    break

                state_estimator.update()
                renderer.update_scene(d, camera=camera_id)
                image = renderer.render()

                cmd = controller.predict_control(image)
                controller.apply_joystick_commands(cmd)
                d.ctrl[:4] = controller.compute_control()

                if video_recorder is not None:
                    pos = state_estimator.base_pos
                    text_lines = [
                        f"Time: {d.time:.2f}s Step: {step_count}",
                        f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]",
                        f"Cmd: [{cmd[0]:.3f}, {cmd[1]:.3f}, {cmd[2]:.3f}, {cmd[3]:.3f}]"
                    ]
                    video_recorder.add_frame(add_text_overlay(image, text_lines))

                mujoco.mj_step(m, d)
                viewer.sync()
                step_count += 1

                if time.time() - last_print_time > 2.0:
                    print(f"Step {step_count} | Pos: {state_estimator.base_pos}")
                    last_print_time = time.time()

                time.sleep(max(0, m.opt.timestep - (time.time() - last_print_time)))

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            if video_recorder:
                video_recorder.stop()
            stats = controller.get_inference_stats()
            print(f"\n推理统计: 平均 {stats['avg_time']:.2f} ms, FPS {stats['fps']:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--episode_num', type=int, default=60)
    parser.add_argument('--record_video', action='store_true', default=True)
    parser.add_argument('--flow_method', type=str, default='farneback')
    parser.add_argument('--fusion_method', type=str, default='attention')
    args = parser.parse_args()
    main_inference_test(**vars(args))