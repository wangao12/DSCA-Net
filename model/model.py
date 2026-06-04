import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CNNBlock(nn.Module):
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


class DualStreamLSTMDroneModel(nn.Module):
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