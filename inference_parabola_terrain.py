# inference_parabola_DualStreamLSTM.py
# 新模型（双流 LSTM + 光流）在抛物线沟壑地形上的推理测试
# 已适配训练代码 OpticalFlow_LSTMA_v6.py 的模型结构（幅度-方向编码 + 置信度掩码 + 跨模态注意力）

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mujoco
import time
from collections import deque
import matplotlib.pyplot as plt
from parabola_control import Drone, UserCommand  # 请确保此模块可用

# ======================== 光流后处理（从训练代码迁移）========================
def flow_to_magnitude_angle_confidence(flow_uv, conf_threshold_quantile=0.3):
    """
    将 (u,v) 光流转换为：
        magnitude : 归一化幅度
        angle_sin, angle_cos : 方向的正余弦
        confidence : 基于幅度的置信度掩码
    """
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    mag = np.sqrt(u**2 + v**2) + 1e-8

    # 幅度归一化（此处返回原始 mag，后续预处理时会除以图像尺寸）
    mag_norm = mag

    # 方向 sin/cos
    sin_angle = v / mag
    cos_angle = u / mag

    # 置信度：基于幅度的指数函数，认为较大运动更可靠
    mag_thresh = np.quantile(mag, conf_threshold_quantile)
    confidence = 1.0 - np.exp(-mag / (mag_thresh + 1e-8))
    confidence = np.clip(confidence, 0.1, 1.0)  # 避免过小

    mag_ang = np.stack([mag, sin_angle, cos_angle], axis=-1)  # (H, W, 3)
    confidence = confidence[..., np.newaxis]  # (H, W, 1)

    return mag_ang, confidence


# ======================== 模型定义（与训练代码完全一致）========================
class CNNBlock(nn.Module):
    """标准卷积块：Conv + BN + ReLU + MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
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
    """增强视觉编码器，支持返回中间特征图"""
    def __init__(self, image_size=(240, 320), in_channels=3):
        super(EnhancedVisionEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.block1 = CNNBlock(32, 64)
        self.block2 = CNNBlock(64, 128)
        self.block3 = CNNBlock(128, 256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 4))  # 输出: 256, 3, 4

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


class CrossModalAttention(nn.Module):
    """交叉模态注意力：光流引导RGB，RGB修正光流"""
    def __init__(self, rgb_channels, flow_channels, reduction=8):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.flow_channels = flow_channels

        # 生成 Query/Key/Value 的投影层（1x1卷积）
        self.rgb_query = nn.Conv2d(rgb_channels, rgb_channels // reduction, 1)
        self.flow_key = nn.Conv2d(flow_channels, rgb_channels // reduction, 1)
        self.flow_value = nn.Conv2d(flow_channels, rgb_channels, 1)

        self.flow_query = nn.Conv2d(flow_channels, flow_channels // reduction, 1)
        self.rgb_key = nn.Conv2d(rgb_channels, flow_channels // reduction, 1)
        self.rgb_value = nn.Conv2d(rgb_channels, flow_channels, 1)

        # 输出投影
        self.rgb_out = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 1),
            nn.BatchNorm2d(rgb_channels)
        )
        self.flow_out = nn.Sequential(
            nn.Conv2d(flow_channels, flow_channels, 1),
            nn.BatchNorm2d(flow_channels)
        )

    def forward(self, rgb_feat, flow_feat):
        """
        rgb_feat: (B, C_r, H, W)
        flow_feat: (B, C_f, H, W)
        返回增强后的特征图
        """
        B, _, H, W = rgb_feat.shape

        # ---- 光流引导 RGB ----
        Q_r = self.rgb_query(rgb_feat).view(B, -1, H*W)        # (B, C_r//r, N)
        K_f = self.flow_key(flow_feat).view(B, -1, H*W)        # (B, C_r//r, N)
        V_f = self.flow_value(flow_feat).view(B, -1, H*W)      # (B, C_r, N)

        attn_r = F.softmax(torch.bmm(Q_r.transpose(1,2), K_f), dim=-1)  # (B, N, N)
        rgb_attended = torch.bmm(V_f, attn_r.transpose(1,2)).view(B, -1, H, W)  # (B, C_r, H, W)
        rgb_enhanced = self.rgb_out(rgb_feat + rgb_attended)

        # ---- RGB 修正光流 ----
        Q_f = self.flow_query(flow_feat).view(B, -1, H*W)
        K_r = self.rgb_key(rgb_feat).view(B, -1, H*W)
        V_r = self.rgb_value(rgb_feat).view(B, -1, H*W)

        attn_f = F.softmax(torch.bmm(Q_f.transpose(1,2), K_r), dim=-1)
        flow_attended = torch.bmm(V_r, attn_f.transpose(1,2)).view(B, -1, H, W)
        flow_enhanced = self.flow_out(flow_feat + flow_attended)

        return rgb_enhanced, flow_enhanced


class DualStreamLSTMDroneModel(nn.Module):
    """双流 LSTM 无人机模型：RGB 和光流分别编码，分别 LSTM，然后融合（通道门控网络）"""
    def __init__(self, sequence_length=5, joystick_command_dim=4,
                 image_size=(240, 320), fusion_method='concat'):
        super(DualStreamLSTMDroneModel, self).__init__()
        self.sequence_length = sequence_length
        self.fusion_method = fusion_method

        # RGB 编码器（输入 3 通道）
        self.rgb_encoder = EnhancedVisionEncoder(image_size=image_size, in_channels=3)
        # 光流编码器（输入 3 通道：mag, sin, cos）
        self.flow_encoder = EnhancedVisionEncoder(image_size=image_size, in_channels=3)
        rgb_feat_dim = 2048
        flow_feat_dim = 2048

        # 跨模态注意力（作用在 intermediate 特征图上，尺寸 3x4）
        self.cross_attn = CrossModalAttention(rgb_channels=256, flow_channels=256, reduction=8)

        # 置信度下采样模块（用于加权光流特征）
        self.conf_downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 4)),   # 与中间特征图尺寸匹配
        )

        # LSTM
        self.rgb_lstm = nn.LSTM(
            rgb_feat_dim, 128, batch_first=True,
            num_layers=2, dropout=0.3, bidirectional=False
        )
        self.flow_lstm = nn.LSTM(
            flow_feat_dim, 64, batch_first=True,
            num_layers=2, dropout=0.3, bidirectional=False
        )

        # 运动增强
        self.motion_enhancer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
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
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, joystick_command_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        if self.channel_gate is not None:
            last_linear = self.channel_gate[-2]
            if isinstance(last_linear, nn.Linear):
                nn.init.constant_(last_linear.bias, 1.0)

    def forward(self, rgb, flow_mag_ang, flow_conf):
        """
        rgb: (B, T, 3, H, W)
        flow_mag_ang: (B, T, 3, H, W)
        flow_conf: (B, T, 1, H, W)
        """
        batch_size, seq_len = rgb.shape[0], rgb.shape[1]

        # 展平时间维度
        rgb_flat = rgb.reshape(batch_size * seq_len, *rgb.shape[2:])
        flow_flat = flow_mag_ang.reshape(batch_size * seq_len, *flow_mag_ang.shape[2:])
        conf_flat = flow_conf.reshape(batch_size * seq_len, *flow_conf.shape[2:])

        # 编码器提取中间特征图 + 最终向量
        rgb_feat_vec, rgb_inter = self.rgb_encoder(rgb_flat, return_intermediate=True)
        flow_feat_vec, flow_inter = self.flow_encoder(flow_flat, return_intermediate=True)

        # 跨模态注意力增强中间特征图
        rgb_inter_enh, flow_inter_enh = self.cross_attn(rgb_inter, flow_inter)

        # 置信度下采样并加权光流中间特征图
        conf_down = self.conf_downsample(conf_flat)  # (B*T, 1, 3, 4)
        flow_inter_weighted = flow_inter_enh * conf_down

        # 将增强后的中间特征图送入压缩模块，得到最终向量
        rgb_feat_vec_final = self.rgb_encoder.feature_compress(rgb_inter_enh)
        flow_feat_vec_final = self.flow_encoder.feature_compress(flow_inter_weighted)

        # 恢复时间维度
        rgb_feat_seq = rgb_feat_vec_final.reshape(batch_size, seq_len, -1)
        flow_feat_seq = flow_feat_vec_final.reshape(batch_size, seq_len, -1)

        # LSTM
        rgb_lstm_out, _ = self.rgb_lstm(rgb_feat_seq)
        flow_lstm_out, _ = self.flow_lstm(flow_feat_seq)
        rgb_last = rgb_lstm_out[:, -1, :]   # (B, 128)
        flow_last = flow_lstm_out[:, -1, :] # (B, 64)

        # 运动增强
        enhanced_flow = self.motion_enhancer(flow_last)

        # 融合
        combined = torch.cat([rgb_last, enhanced_flow], dim=1)
        if self.fusion_method == 'attention' and self.channel_gate is not None:
            gate = self.channel_gate(combined)
            combined_weighted = combined * gate
            fused = self.fusion(combined_weighted)
        else:
            fused = self.fusion(combined)

        joystick_command = self.joystick_command_predictor(fused)
        return joystick_command


# ======================== 光流计算工具类 ========================
class OpticalFlowCalculator:
    def __init__(self, method='farneback'):
        self.method = method
        self.prev_gray = None

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
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    flow[int(b), int(a)] = [a - c, b - d]
                flow[:, :, 0] = cv2.medianBlur(flow[:, :, 0], 5)
                flow[:, :, 1] = cv2.medianBlur(flow[:, :, 1], 5)
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
        self.frame_count = 0

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, self.frame_size
        )
        self.frame_count = 0
        print(f"开始录制视频: {self.output_path}")

    def add_frame(self, frame):
        if self.video_writer is not None:
            if frame.shape[:2] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)
            self.frame_count += 1

    def stop(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"视频录制完成: {self.output_path} (共 {self.frame_count} 帧)")
            self.video_writer = None


# ======================== 推理控制器（适配新模型）========================
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

        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()

        # 加载归一化参数（只有手柄命令的均值和标准差）
        self.normalization_params = self.load_normalization_params(normalization_path)

        # 初始化抛物线Drone控制器
        self.drone_controller = Drone()
        self.user_cmd = UserCommand()

        # 序列数据存储（图像、光流幅角、置信度）
        self.image_sequence = deque(maxlen=sequence_length)
        self.flow_mag_ang_sequence = deque(maxlen=sequence_length)
        self.flow_conf_sequence = deque(maxlen=sequence_length)
        self.prev_image = None

        # 状态跟踪（仅用于记录和显示，不输入网络）
        self.state_history = deque(maxlen=100)
        self.command_history = deque(maxlen=100)

        # 推理时间统计
        self.inference_times = []
        self.total_inference_time = 0.0

        print(f"双流 LSTM 推理控制器初始化完成 (序列长度: {sequence_length}, 光流方法: {flow_method}, 融合方法: {fusion_method})")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        sequence_length = config.get('sequence_length', 5)
        joystick_command_dim = config.get('joystick_command_dim', 4)
        image_size = config.get('image_size', (240, 320))
        fusion_method = config.get('fusion_method', 'attention')

        model = DualStreamLSTMDroneModel(
            sequence_length=sequence_length,
            joystick_command_dim=joystick_command_dim,
            image_size=image_size,
            fusion_method=fusion_method
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        print(f"双流 LSTM 模型已加载: {model_path}")
        print(f"序列长度: {sequence_length}, 手柄命令维度: {joystick_command_dim}")
        print(f"光流方法: {self.flow_method}, 融合方法: {fusion_method}")
        return model

    def load_normalization_params(self, normalization_path):
        params = np.load(normalization_path, allow_pickle=True).item()
        if 'joystick_command_mean' not in params:
            raise KeyError("归一化文件缺少 joystick_command_mean，请确认使用新模型训练产生的归一化文件")
        print("归一化参数已加载 (仅手柄命令)")
        return params

    def preprocess_image(self, image):
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(image).float()

    def preprocess_flow(self, flow_uv):
        """将原始 (u,v) 光流转换为幅度-方向编码和置信度，并调整尺寸"""
        if flow_uv.shape[:2] != self.image_size:
            flow_uv = cv2.resize(flow_uv, (self.image_size[1], self.image_size[0]))
        # 转换为幅角 + 置信度
        mag_ang, conf = flow_to_magnitude_angle_confidence(flow_uv)
        # 调整尺寸（mag_ang 和 conf 都是 ndarray）
        if mag_ang.shape[:2] != self.image_size:
            mag_ang = cv2.resize(mag_ang, (self.image_size[1], self.image_size[0]))
            conf = cv2.resize(conf, (self.image_size[1], self.image_size[0]))
        # 转置为 CHW
        mag_ang = mag_ang.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)
        conf = conf.transpose(2, 0, 1).astype(np.float32)        # (1, H, W)
        # 幅度归一化（除以图像尺寸，与训练时一致）
        mag_ang[0] /= max(self.image_size[1], self.image_size[0])
        return torch.from_numpy(mag_ang).float(), torch.from_numpy(conf).float()

    def postprocess_command(self, joystick_command):
        mean = self.normalization_params['joystick_command_mean']
        std = self.normalization_params['joystick_command_std']
        return joystick_command * std + mean

    def add_to_sequence(self, image):
        processed_image = self.preprocess_image(image)

        # 计算光流并转换为幅角+置信度
        if self.prev_image is not None:
            flow_uv = self.flow_calculator.calculate_flow(self.prev_image, image)
            mag_ang, conf = self.preprocess_flow(flow_uv)
        else:
            # 第一帧：零光流 -> 零幅度 -> 方向为(0,0)，置信度为0
            zero_flow_uv = np.zeros((self.image_size[0], self.image_size[1], 2), dtype=np.float32)
            mag_ang, conf = self.preprocess_flow(zero_flow_uv)

        self.prev_image = image.copy()
        self.image_sequence.append(processed_image)
        self.flow_mag_ang_sequence.append(mag_ang)
        self.flow_conf_sequence.append(conf)

        # 如果序列长度不足，重复第一帧
        while len(self.image_sequence) < self.sequence_length:
            self.image_sequence.appendleft(processed_image)
            self.flow_mag_ang_sequence.appendleft(mag_ang)
            self.flow_conf_sequence.appendleft(conf)

    def get_sequence_tensors(self):
        if len(self.image_sequence) < self.sequence_length:
            raise ValueError(f"序列长度不足: {len(self.image_sequence)} < {self.sequence_length}")
        image_tensor = torch.stack(list(self.image_sequence)).unsqueeze(0).to(self.device)
        flow_mag_ang_tensor = torch.stack(list(self.flow_mag_ang_sequence)).unsqueeze(0).to(self.device)
        flow_conf_tensor = torch.stack(list(self.flow_conf_sequence)).unsqueeze(0).to(self.device)
        return image_tensor, flow_mag_ang_tensor, flow_conf_tensor

    def predict_control(self, image):
        self.add_to_sequence(image)

        with torch.no_grad():
            image_tensor, flow_mag_ang_tensor, flow_conf_tensor = self.get_sequence_tensors()
            start_time = time.time()
            joystick_command_normalized = self.model(image_tensor, flow_mag_ang_tensor, flow_conf_tensor)
            inference_time = time.time() - start_time

            self.inference_times.append(inference_time)
            self.total_inference_time += inference_time

            joystick_command = joystick_command_normalized.cpu().numpy()[0]
            joystick_command_denormalized = self.postprocess_command(joystick_command)
            self.command_history.append(joystick_command_denormalized.copy())
            return joystick_command_denormalized

    def get_inference_stats(self):
        if not self.inference_times:
            return {'avg_time': 0, 'min_time': 0, 'max_time': 0, 'total_time': 0, 'num_inferences': 0, 'fps': 0}
        return {
            'avg_time': np.mean(self.inference_times) * 1000,
            'min_time': np.min(self.inference_times) * 1000,
            'max_time': np.max(self.inference_times) * 1000,
            'total_time': self.total_inference_time,
            'num_inferences': len(self.inference_times),
            'fps': len(self.inference_times) / self.total_inference_time if self.total_inference_time > 0 else 0
        }

    def apply_joystick_commands(self, joystick_command):
        v_x_local, v_y_local, v_z, yaw_rate = joystick_command
        self.drone_controller.v_x_local_processed = v_x_local
        self.drone_controller.v_y_local_processed = v_y_local
        self.drone_controller.v_z_processed = v_z
        self.drone_controller.yaw_rate_processed = yaw_rate
        self._update_user_command_from_neural_network(v_x_local, v_y_local, v_z, yaw_rate)

    def _update_user_command_from_neural_network(self, v_x_local, v_y_local, v_z, yaw_rate):
        self.user_cmd.yaw += yaw_rate * self.drone_controller.YAW_RATE_SCALE * self.drone_controller.m.opt.timestep
        current_yaw = self.user_cmd.yaw
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        v_x_world = (v_x_local * cos_yaw - v_y_local * sin_yaw) * self.drone_controller.XY_VEL_SCALE
        v_y_world = (v_x_local * sin_yaw + v_y_local * cos_yaw) * self.drone_controller.XY_VEL_SCALE
        self.user_cmd.x += v_x_world * self.drone_controller.m.opt.timestep
        self.user_cmd.y += v_y_world * self.drone_controller.m.opt.timestep
        self.user_cmd.z += v_z * self.drone_controller.Z_VEL_SCALE * self.drone_controller.m.opt.timestep
        self.user_cmd.z = np.clip(self.user_cmd.z, 0.2, 2.5)

        drone_pos = self.drone_controller.state_estimator.pos
        target_pos = np.array([self.user_cmd.x, self.user_cmd.y, self.user_cmd.z])
        error_vec = target_pos - drone_pos
        distance = np.linalg.norm(error_vec)
        if distance > self.drone_controller.MAX_TARGET_DISTANCE:
            clamped_error = error_vec / distance * self.drone_controller.MAX_TARGET_DISTANCE
            new_target_pos = drone_pos + clamped_error
            self.user_cmd.x, self.user_cmd.y, self.user_cmd.z = new_target_pos

    def compute_control(self):
        self.drone_controller.user_cmd = self.user_cmd
        self.drone_controller()
        return self.drone_controller.d.ctrl[:4].copy()


# ======================== 状态估计器（用于控制）========================
class InferenceStateEstimator:
    def __init__(self, m, d):
        self.m = m
        self.d = d
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.base_vel_lin_global = np.zeros(3)
        self.base_vel_ang_local = np.zeros(3)
        self.pos = np.zeros(3)
        self.R = np.eye(3)

    def update(self):
        self.base_pos = self.d.qpos[:3].copy()
        self.base_quat = self.d.qpos[3:7].copy()
        self.pos = self.base_pos.copy()
        self.base_vel_lin_global = self.d.qvel[:3].copy()
        self.base_vel_ang_local = self.d.qvel[3:6].copy()
        w, x, y, z = self.base_quat
        self.R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])


# ======================== 辅助函数 ========================
def add_text_overlay(image, text_lines):
    overlay = image.copy()
    text_bg_height = len(text_lines) * 30 + 10
    if text_bg_height > overlay.shape[0]:
        max_lines = (overlay.shape[0] - 10) // 30
        if max_lines <= 0:
            max_lines = 1
        text_lines = text_lines[:max_lines]
        text_bg_height = max_lines * 30 + 10
    h, w = overlay.shape[:2]
    text_bg = np.zeros((text_bg_height, w, 3), dtype=np.uint8)
    text_bg[:, :] = (0, 0, 0)
    if text_bg_height <= h:
        overlay[0:text_bg_height, 0:w] = cv2.addWeighted(overlay[0:text_bg_height, 0:w], 0.3, text_bg, 0.7, 0)
    for i, line in enumerate(text_lines):
        if i * 30 + 25 < h:
            cv2.putText(overlay, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def visualize_optical_flow_for_video(mag_ang, frame_size=(320, 240)):
    """从 mag_ang (C, H, W) 数组中可视化光流，使用幅度-角度图"""
    # mag_ang shape: (3, H, W)，索引0是幅度，1是sin，2是cos
    mag = mag_ang[0, :, :]
    sin_a = mag_ang[1, :, :]
    cos_a = mag_ang[2, :, :]
    # 计算角度（转为度）
    angle = np.arctan2(sin_a, cos_a) * 180 / np.pi
    angle = (angle + 180) / 2  # 归一化到0-180

    hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle.astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_rgb = cv2.resize(flow_rgb, frame_size)
    return flow_rgb


# ======================== 主推理函数（抛物线地形）========================
def main_inference_test(model_dir=None, episode_num=60, record_video=True,
                        flow_method='farneback', fusion_method='attention'):
    # 默认模型目录名（与训练脚本输出一致）
    if model_dir is None:
        model_dir = f"DualStream_Improved_{episode_num}_ep_seq5_ds1_flow_{flow_method}_fusion_{fusion_method}"

    model_path = os.path.join(model_dir, "best_model.pth")
    normalization_path = os.path.join(model_dir, "normalization_params.npy")

    if not os.path.exists(model_path) or not os.path.exists(normalization_path):
        print(f"错误: 找不到模型文件或归一化文件")
        print(f"模型路径: {model_path}")
        print(f"归一化路径: {normalization_path}")
        return

    # 从 checkpoint 读取序列长度
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    sequence_length = config.get('sequence_length', 5)
    fusion_method = config.get('fusion_method', fusion_method)

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

    # 设置抛物线沟壑地形
    controller.drone_controller.setup_parabola_trench_terrain()
    if controller.drone_controller.finish_zone_pos is not None:
        controller.drone_controller.finish_zone_pos[2] = max(controller.drone_controller.finish_zone_pos[2], 1.5)

    # 从抛物线参数中获取入口位置
    trench_params = controller.drone_controller.trench_params
    trench_start_x = trench_params['trench_start_x']           # 沟壑入口 x 坐标
    parabola_a = trench_params['parabola_a']
    parabola_h = trench_params['parabola_h']
    parabola_k = trench_params['parabola_k']
    entrance_center_y = parabola_a * (trench_start_x - parabola_h)**2 + parabola_k
    entrance_height = 1.5

    # 设置无人机初始位置
    d.qpos[:3] = [trench_start_x, entrance_center_y, entrance_height]
    d.qpos[3:7] = [1, 0, 0, 0]
    controller.user_cmd.x = trench_start_x
    controller.user_cmd.y = entrance_center_y
    controller.user_cmd.z = entrance_height

    # 视频输出目录
    video_dir = "inference_videos"
    os.makedirs(video_dir, exist_ok=True)

    # 初始化摄像头
    renderer = mujoco.Renderer(m, 240, 320)
    camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "drone_eye")
    if camera_id < 0:
        camera_id = 0

    # 视频录制
    video_recorder = None
    flow_video_recorder = None
    if record_video:
        video_path = os.path.join(video_dir, f"inference_DualStream_parabola_ep{episode_num}.mp4")
        video_recorder = VideoRecorder(video_path, fps=30, frame_size=(320, 240))
        video_recorder.start()
        flow_video_path = os.path.join(video_dir, f"optical_flow_parabola_ep{episode_num}.mp4")
        flow_video_recorder = VideoRecorder(flow_video_path, fps=30, frame_size=(320, 240))
        flow_video_recorder.start()

    print("=" * 60)
    print("双流 LSTM (RGB+光流) 模型在抛物线沟壑地形上的推理测试")
    print("=" * 60)
    print(f"序列长度: {sequence_length}, 光流方法: {flow_method}, 融合方法: {fusion_method}")
    print(f"抛物线参数: a={parabola_a:.4f}, h={parabola_h:.2f}, k={parabola_k:.2f}")
    print(f"沟壑入口位置: x={trench_start_x:.2f}, y={entrance_center_y:.2f}")
    print(f"无人机初始位置: [{trench_start_x:.2f}, {entrance_center_y:.2f}, {entrance_height:.2f}]")
    if controller.drone_controller.finish_zone_pos is not None:
        finish_pos = controller.drone_controller.finish_zone_pos
        print(f"终点位置: [{finish_pos[0]:.2f}, {finish_pos[1]:.2f}, {finish_pos[2]:.2f}]")
        print(f"任务目标: 飞向抛物线沟壑终点区域（半径 {controller.drone_controller.FINISH_ZONE_RADIUS}m）")
    print("=" * 60)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
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
                step_start = time.time()
                if controller.drone_controller.mission_complete:
                    mission_complete = True
                    print("🎉 任务已完成！")
                    break

                state_estimator.update()
                drone_pos = state_estimator.base_pos

                renderer.update_scene(d, camera=camera_id)
                image = renderer.render()

                # 模型预测（只输入图像）
                joystick_command = controller.predict_control(image)
                controller.apply_joystick_commands(joystick_command)
                motor_commands = controller.compute_control()
                d.ctrl[:4] = motor_commands

                # 视频录制和文本叠加
                if video_recorder is not None:
                    current_flow_mag = "N/A"
                    if len(controller.flow_mag_ang_sequence) > 0:
                        latest_mag_ang = controller.flow_mag_ang_sequence[-1]
                        if hasattr(latest_mag_ang, 'cpu'):
                            latest_mag_ang = latest_mag_ang.cpu().numpy()
                        # mag_ang 形状 (3, H, W)，取第0通道的均值
                        current_flow_mag = f"{latest_mag_ang[0].mean():.3f}"

                    text_lines = [
                        f"Time: {d.time:.2f}s",
                        f"Step: {step_count}",
                        f"Sequence: {len(controller.image_sequence)}/{sequence_length}",
                        f"Position: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}]",
                        f"Commands: [{joystick_command[0]:.3f}, {joystick_command[1]:.3f}, {joystick_command[2]:.3f}, {joystick_command[3]:.3f}]",
                        f"Flow Mag: {current_flow_mag}"
                    ]
                    if controller.drone_controller.finish_zone_pos is not None:
                        distance_to_finish = np.linalg.norm(drone_pos - controller.drone_controller.finish_zone_pos)
                        text_lines.append(f"Dist to finish: {distance_to_finish:.2f}m")

                    video_frame = add_text_overlay(image, text_lines)
                    video_recorder.add_frame(video_frame)

                    if flow_video_recorder is not None and len(controller.flow_mag_ang_sequence) > 0:
                        latest_mag_ang = controller.flow_mag_ang_sequence[-1]
                        if hasattr(latest_mag_ang, 'cpu'):
                            latest_mag_ang = latest_mag_ang.cpu().numpy()
                        # 可视化 mag_ang
                        flow_vis = visualize_optical_flow_for_video(latest_mag_ang)
                        flow_video_recorder.add_frame(flow_vis)

                mujoco.mj_step(m, d)
                viewer.sync()
                step_count += 1

                if time.time() - last_print_time > 2.0:
                    if controller.drone_controller.finish_zone_pos is not None:
                        distance_to_finish = np.linalg.norm(drone_pos - controller.drone_controller.finish_zone_pos)
                        print(f"Time: {d.time:.2f}s | Steps: {step_count} | Pos: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}] | Dist to finish: {distance_to_finish:.2f}m")
                    else:
                        print(f"Time: {d.time:.2f}s | Steps: {step_count} | Pos: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}]")
                    last_print_time = time.time()

                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        except KeyboardInterrupt:
            print("\n用户中断推理测试")
        finally:
            if video_recorder is not None:
                video_recorder.stop()
            if flow_video_recorder is not None:
                flow_video_recorder.stop()

            inference_stats = controller.get_inference_stats()
            print("\n" + "=" * 60)
            print("推理性能统计 (双流 LSTM 模型 - 抛物线沟壑地形):")
            print("=" * 60)
            print(f"模型类型: DualStreamLSTM ({fusion_method}融合)")
            print(f"光流方法: {flow_method}")
            print(f"总推理次数: {inference_stats['num_inferences']}")
            print(f"总推理时间: {inference_stats['total_time']:.3f} 秒")
            print(f"平均推理时间: {inference_stats['avg_time']:.2f} 毫秒")
            print(f"最快/最慢: {inference_stats['min_time']:.2f}/{inference_stats['max_time']:.2f} 毫秒")
            print(f"推理帧率: {inference_stats['fps']:.2f} FPS")
            print(f"仿真总步数: {step_count}, 总时间: {d.time:.2f} 秒")
            print("=" * 60)

            # 保存结果
            test_results = {
                'total_steps': step_count,
                'simulation_time': d.time,
                'final_position': state_estimator.base_pos.copy(),
                'trench_params': controller.drone_controller.trench_params,
                'command_history': list(controller.command_history),
                'mission_complete': mission_complete,
                'sequence_length': sequence_length,
                'inference_stats': inference_stats,
                'model_type': 'DualStreamLSTM_Parabola',
                'fusion_method': fusion_method,
                'flow_method': flow_method
            }
            results_file = os.path.join(video_dir, f"inference_DualStream_parabola_results_ep{episode_num}.npz")
            np.savez(results_file, **test_results)
            print(f"测试结果已保存: {results_file}")
            plot_inference_results(test_results)


def plot_inference_results(results):
    command_history = np.array(results['command_history'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    steps = range(len(command_history))
    axes[0].plot(steps, command_history[:, 0], 'r-', label='v_x_local')
    axes[0].plot(steps, command_history[:, 1], 'g-', label='v_y_local')
    axes[0].plot(steps, command_history[:, 2], 'b-', label='v_z')
    axes[0].plot(steps, command_history[:, 3], 'm-', label='yaw_rate')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Command Value')
    axes[0].set_title('Joystick Commands')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].hist(command_history[:, 0], bins=30, alpha=0.5, label='v_x')
    axes[1].hist(command_history[:, 1], bins=30, alpha=0.5, label='v_y')
    axes[1].hist(command_history[:, 2], bins=30, alpha=0.5, label='v_z')
    axes[1].set_xlabel('Command Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Command Distribution')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('DualStream LSTM on Parabola Trench', fontsize=14)
    plt.tight_layout()
    plt.savefig('inference_DualStream_parabola_results.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='改进版双流 LSTM 无人机在抛物线沟壑地形上的推理测试')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='模型目录路径（默认根据 episode_num, flow_method, fusion_method 自动生成）')
    parser.add_argument('--episode_num', type=int, default=60,
                        help='训练时使用的 episode 数量（用于自动生成模型目录名）')
    parser.add_argument('--record_video', action='store_true', default=True,
                        help='是否录制测试视频')
    parser.add_argument('--flow_method', type=str, default='farneback',
                        choices=['farneback', 'lk'], help='光流计算方法')
    parser.add_argument('--fusion_method', type=str, default='attention',
                        choices=['concat', 'attention'], help='特征融合方法')
    args = parser.parse_args()

    main_inference_test(
        model_dir=args.model_dir,
        episode_num=args.episode_num,
        record_video=args.record_video,
        flow_method=args.flow_method,
        fusion_method=args.fusion_method
    )