"""
条件噪声预测器模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..utils.point_encoder import PointEncoder


class CondNoisePredictor(nn.Module):
    """
    条件噪声预测器
    """
    
    def __init__(self, feat_dim: int = 5, emb_dim: int = 128, grid_size: int = 32):
        """
        初始化模型
        
        Args:
            feat_dim: 特征维度
            emb_dim: 嵌入维度
            grid_size: 网格大小
        """
        super().__init__()
        self.grid_size = grid_size
        self.output_dim = grid_size * grid_size * 3  # 32*32*3 = 3072
        
        # 点云编码器
        self.encoder = PointEncoder(feat_dim, emb_dim)
        
        # 时间嵌入
        self.time_fc = nn.Linear(1, emb_dim)
        
        # 条件嵌入
        self.cond_fc = nn.Linear(3, emb_dim)
        
        # 主网络
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + self.output_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, self.output_dim)
        )
    
    def forward(self, pts: torch.Tensor, noisy_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pts: 点云特征 (B, N, F)
            noisy_points: 噪声点云 (B, grid_size*grid_size*3)
            t: 时间步 (B,)
        
        Returns:
            predicted_noise: 预测的噪声 (B, grid_size*grid_size*3)
        """
        # 编码点云
        cond_raw = self.encoder(pts)           # (B, 3)
        cond_emb = self.cond_fc(cond_raw)      # (B, emb_dim)
        
        # 时间嵌入
        time_emb = torch.sin(self.time_fc(t.float().unsqueeze(1)))  # (B, emb_dim)
        
        # 组合特征
        h = torch.cat([cond_emb + time_emb, noisy_points], dim=1)
        
        # 预测噪声
        return self.mlp(h)
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            model_info: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'grid_size': self.grid_size,
            'output_dim': self.output_dim,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        } 