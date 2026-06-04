import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cv2
import argparse

from model import DualStreamLSTMDroneModel

# ======================== 光流计算工具类 ========================
class OpticalFlowCalculator:
    """光流计算工具类（返回原始 (u,v) 光流）"""
    def __init__(self, method='farneback', device='cpu'):
        self.method = method
        self.device = device

    def calculate_flow(self, img1, img2):
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()

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

    def calculate_flow_sequence(self, images):
        if len(images.shape) == 4:
            if images.shape[-1] == 3:
                pass
            elif images.shape[1] == 3:
                images = images.transpose(0, 2, 3, 1)
        T = len(images)
        flows = []
        for t in range(T - 1):
            flow = self.calculate_flow(images[t], images[t+1])
            flows.append(flow)
        flows.append(np.zeros_like(flows[-1]))
        return np.array(flows)


# ======================== 光流后处理：幅度-方向 + 置信度 ========================
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
    
    mag_norm = mag  # 外部会进行归一化

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


# ======================== 数据集 ========================
class SequenceDroneDataset(Dataset):
    """序列化无人机数据集，返回 RGB、光流幅角（3通道）和置信度掩码"""
    def __init__(self, data_files, sequence_length=5, image_size=(240, 320),
                 downsample_factor=1, use_optical_flow=True,
                 flow_method='farneback', precompute_flow=True):
        self.data_files = data_files
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.downsample_factor = downsample_factor
        self.use_optical_flow = use_optical_flow
        self.flow_method = flow_method
        self.precompute_flow = precompute_flow
        
        if use_optical_flow:
            self.flow_calculator = OpticalFlowCalculator(method=flow_method)
        else:
            self.flow_calculator = None

        # 加载所有数据
        print("加载数据文件... ")
        self.episodes = []
        self.episode_flow_mag_ang = [] if (use_optical_flow and precompute_flow) else None
        self.episode_flow_conf = [] if (use_optical_flow and precompute_flow) else None

        for file_path in data_files:
            with h5py.File(file_path, 'r') as f:
                images = f['images'][:]
                joystick_commands_local = f['joystick_commands_local'][:]

                if downsample_factor > 1:
                    original_length = len(images)
                    images = images[::downsample_factor]
                    joystick_commands_local = joystick_commands_local[::downsample_factor]
                    print(f"下采样: 从 {original_length} 帧减少到 {len(images)} 帧 ")

                if use_optical_flow and precompute_flow:
                    print(f"预计算 {file_path} 的光流（幅角+置信度）... ")
                    flows_uv = self.flow_calculator.calculate_flow_sequence(images)
                    mag_ang_list = []
                    conf_list = []
                    for f_uv in flows_uv:
                        mag_ang, conf = flow_to_magnitude_angle_confidence(f_uv)
                        mag_ang_list.append(mag_ang)
                        conf_list.append(conf)
                    self.episode_flow_mag_ang.append(np.array(mag_ang_list))
                    self.episode_flow_conf.append(np.array(conf_list))

                episode_data = {
                    'images': images,
                    'joystick_commands_local': joystick_commands_local
                }
                self.episodes.append(episode_data)
                print(f"从 {file_path} 加载了 {len(images)} 个数据点 ")

        # 计算手柄命令的全局归一化参数
        all_joystick_commands = np.concatenate([ep['joystick_commands_local'] for ep in self.episodes], axis=0)
        self.joystick_command_mean = np.mean(all_joystick_commands, axis=0)
        self.joystick_command_std = np.std(all_joystick_commands, axis=0) + 1e-8

        for episode in self.episodes:
            episode['joystick_commands_local'] = (episode['joystick_commands_local'] - self.joystick_command_mean) / self.joystick_command_std

        # 生成序列索引
        self.sequence_indices = []
        for episode_idx, episode in enumerate(self.episodes):
            num_frames = len(episode['images'])
            for start_idx in range(num_frames - sequence_length + 1):
                self.sequence_indices.append((episode_idx, start_idx))

        print(f"总序列数: {len(self.sequence_indices)} ")
        print(f"每序列长度: {sequence_length} ")
        print(f"使用光流: {use_optical_flow}, 方法: {flow_method}, 预计算: {precompute_flow} ")

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        episode_idx, start_idx = self.sequence_indices[idx]
        episode = self.episodes[episode_idx]

        rgb_sequence = []
        flow_mag_ang_sequence = []
        flow_conf_sequence = []
        joystick_command_sequence = []

        for i in range(self.sequence_length):
            frame_idx = start_idx + i
            image = episode['images'][frame_idx]
            if image.shape[:2] != self.image_size:
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            rgb_sequence.append(image)

            if self.use_optical_flow:
                if i < self.sequence_length - 1:
                    if self.precompute_flow and self.episode_flow_mag_ang is not None:
                        mag_ang = self.episode_flow_mag_ang[episode_idx][frame_idx]
                        conf = self.episode_flow_conf[episode_idx][frame_idx]
                    else:
                        # 实时计算（不推荐，但保留）
                        next_img = episode['images'][frame_idx+1]
                        if next_img.shape[:2] != self.image_size:
                            next_img = cv2.resize(next_img, (self.image_size[1], self.image_size[0]))
                        flow_uv = self.flow_calculator.calculate_flow(image.transpose(1,2,0), next_img)
                        mag_ang, conf = flow_to_magnitude_angle_confidence(flow_uv)
                    
                    # 调整尺寸
                    if mag_ang.shape[:2] != self.image_size:
                        mag_ang = cv2.resize(mag_ang, (self.image_size[1], self.image_size[0]))
                        conf = cv2.resize(conf, (self.image_size[1], self.image_size[0]))
                    mag_ang = mag_ang.transpose(2, 0, 1).astype(np.float32)  # CHW
                    conf = conf.transpose(2, 0, 1).astype(np.float32)        # CHW
                    # 幅度归一化（除以图像尺寸）
                    mag_ang[0] /= max(self.image_size[1], self.image_size[0])
                else:
                    # 最后一帧用零填充
                    mag_ang = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=np.float32)
                    conf = np.zeros((1, self.image_size[0], self.image_size[1]), dtype=np.float32)
                flow_mag_ang_sequence.append(mag_ang)
                flow_conf_sequence.append(conf)

            if i == self.sequence_length - 1:
                joystick_command_sequence.append(episode['joystick_commands_local'][frame_idx])

        result = {
            'rgb': torch.from_numpy(np.array(rgb_sequence)).float(),
            'joystick_command': torch.from_numpy(np.array(joystick_command_sequence[0])).float()
        }
        if self.use_optical_flow:
            result['flow_mag_ang'] = torch.from_numpy(np.array(flow_mag_ang_sequence)).float()
            result['flow_conf'] = torch.from_numpy(np.array(flow_conf_sequence)).float()
        return result


# ======================== 早停与训练器 ========================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"验证损失改善: {val_loss:.6f} (最佳)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"验证损失未改善: {val_loss:.6f} (最佳: {self.best_loss:.6f}, 计数: {self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"早停触发! 最佳验证损失: {self.best_loss:.6f}")
        return self.early_stop

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"已加载最佳模型状态 (验证损失: {self.best_loss:.6f})")


class SequenceTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model = DualStreamLSTMDroneModel(
            sequence_length=config['sequence_length'],
            joystick_command_dim=config['joystick_command_dim'],
            image_size=config['image_size'],
            fusion_method=config.get('fusion_method', 'concat')
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        self.early_stopping = EarlyStopping(patience=config.get('early_stopping_patience', 15),
                                             min_delta=config.get('early_stopping_min_delta', 1e-4))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        self.train_history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        os.makedirs(config['output_dir'], exist_ok=True)
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())} ")
        print(f"融合方法: {config.get('fusion_method', 'concat')} ")

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, num_batches = 0, 0
        pbar = tqdm(dataloader, desc="训练 ")
        for batch in pbar:
            rgb = batch['rgb'].to(self.device)
            flow_mag_ang = batch['flow_mag_ang'].to(self.device)
            flow_conf = batch['flow_conf'].to(self.device)
            joystick_commands = batch['joystick_command'].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(rgb, flow_mag_ang, flow_conf)
            loss = self.criterion(pred, joystick_commands)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        return total_loss / num_batches

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss, num_batches = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                rgb = batch['rgb'].to(self.device)
                flow_mag_ang = batch['flow_mag_ang'].to(self.device)
                flow_conf = batch['flow_conf'].to(self.device)
                joystick_commands = batch['joystick_command'].to(self.device)
                pred = self.model(rgb, flow_mag_ang, flow_conf)
                loss = self.criterion(pred, joystick_commands)
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches

    def train(self, train_loader, val_loader):
        print("开始训练... ")
        start_time = time.time()
        best_val_loss = float('inf')
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} ")
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rate'].append(current_lr)
            print(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, 学习率: {current_lr:.2e} ")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                print(f"新的最佳模型已保存! 验证损失: {val_loss:.6f} ")
            if self.early_stopping(val_loss, self.model):
                print(f"早停在 epoch {epoch+1} 触发 ")
                self.early_stopping.load_best_model(self.model)
                break
        training_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {training_time:.2f}秒 ")
        print(f"最佳验证损失: {best_val_loss:.6f} ")
        self.save_model('final_model.pth')
        self.plot_training_history()

    def save_model(self, filename):
        model_path = os.path.join(self.config['output_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_history': self.train_history,
            'config': self.config,
            'early_stopping_best_loss': self.early_stopping.best_loss if hasattr(self, 'early_stopping') else None
        }, model_path)

    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        axes[0].plot(epochs, self.train_history['train_loss'], 'b-', label='训练损失')
        axes[0].plot(epochs, self.train_history['val_loss'], 'r-', label='验证损失')
        axes[0].set_title('训练和验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        if len(self.train_history['val_loss']) > 0:
            best_epoch = np.argmin(self.train_history['val_loss']) + 1
            best_loss = min(self.train_history['val_loss'])
            axes[0].plot(best_epoch, best_loss, 'go', markersize=10, label=f'最佳: {best_loss:.4f}')
            axes[0].legend()
        axes[1].plot(epochs, self.train_history['learning_rate'], 'g-', label='学习率')
        axes[1].set_title('学习率变化')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'training_history.png'), dpi=300)
        plt.show()


# ======================== 数据准备与主函数 ========================
def prepare_sequence_data(data_files, sequence_length=5, batch_size=16,
                          test_size=0.2, downsample_factor=1,
                          use_optical_flow=True, flow_method='farneback',
                          precompute_flow=True):
    dataset = SequenceDroneDataset(
        data_files,
        sequence_length=sequence_length,
        downsample_factor=downsample_factor,
        use_optical_flow=use_optical_flow,
        flow_method=flow_method,
        precompute_flow=precompute_flow
    )
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=test_size, random_state=42, shuffle=True)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"训练集序列数: {len(train_dataset)}")
    print(f"验证集序列数: {len(val_dataset)}")
    return train_loader, val_loader, dataset


def get_data_files(num_episodes, data_dir="v6_expert_data_trench_low_freq"):
    data_files = []
    for i in range(1, num_episodes + 1):
        file_path = os.path.join(data_dir, f"mission_complete_episode_{i}.h5")
        if os.path.exists(file_path):
            data_files.append(file_path)
        else:
            import glob
            file_path = os.path.join(data_dir, f"expert_data_episode_{i:03d}_*.h5")
            matching_files = glob.glob(file_path)
            if matching_files:
                data_files.extend(matching_files)
            else:
                print(f"警告: 数据文件 {file_path} 不存在!")
    return data_files


def main(num_episodes=60, num_epochs=100, sequence_length=5,
         downsample_factor=1, use_optical_flow=True,
         flow_method='farneback', precompute_flow=True,
         fusion_method='concat', data_dir="v6_expert_data_trench_low_freq",
         early_stopping_patience=6, early_stopping_min_delta=1e-4):
    
    data_files = get_data_files(num_episodes, data_dir)
    if len(data_files) == 0:
        print("错误: 没有找到任何数据文件!")
        return
        
    config = {
        'data_files': data_files,
        'output_dir': f'DualStream_Improved_{num_episodes}_ep_seq{sequence_length}_ds{downsample_factor}_flow_{flow_method}_fusion_{fusion_method}',
        'sequence_length': sequence_length,
        'joystick_command_dim': 4,
        'image_size': (240, 320),
        'batch_size': 16,
        'learning_rate': 3e-4,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_min_delta': early_stopping_min_delta,
        'use_optical_flow': use_optical_flow,
        'flow_method': flow_method,
        'precompute_flow': precompute_flow,
        'fusion_method': fusion_method
    }

    print(f"使用 {len(data_files)} 个数据文件训练: ")
    for file in data_files:
        print(f"  - {file} ")
    print(f"序列长度: {sequence_length} ")
    print(f"下采样因子: {downsample_factor} ")
    print(f"使用光流: {use_optical_flow}, 方法: {flow_method}, 预计算: {precompute_flow} ")
    print(f"融合方法: {fusion_method} ")
    print("模型改进: 幅度-方向编码 + 置信度掩码 + 跨模态注意力 ")

    train_loader, val_loader, dataset = prepare_sequence_data(
        config['data_files'],
        sequence_length=config['sequence_length'],
        batch_size=config['batch_size'],
        downsample_factor=downsample_factor,
        use_optical_flow=use_optical_flow,
        flow_method=flow_method,
        precompute_flow=precompute_flow
    )

    trainer = SequenceTrainer(config)
    trainer.train(train_loader, val_loader)

    normalization_params = {
        'joystick_command_mean': dataset.joystick_command_mean,
        'joystick_command_std': dataset.joystick_command_std
    }
    np.save(os.path.join(config['output_dir'], 'normalization_params.npy'), normalization_params)
    print("归一化参数已保存 ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='改进版双流 LSTM 无人机模仿学习（幅角+置信度+跨模态注意力）')
    parser.add_argument('--num_episodes', type=int, default=60)
    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--sequence_length', type=int, default=5)
    parser.add_argument('--downsample_factor', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="v6_expert_data_trench_low_freq")
    parser.add_argument('--early_stopping_patience', type=int, default=6)
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4)
    parser.add_argument('--use_optical_flow', action='store_true', default=True)
    parser.add_argument('--flow_method', type=str, default='farneback', choices=['farneback', 'lk'])
    parser.add_argument('--precompute_flow', action='store_true', default=True)
    parser.add_argument('--fusion_method', type=str, default='attention', choices=['concat', 'attention'])
    args = parser.parse_args()
    main(**vars(args))