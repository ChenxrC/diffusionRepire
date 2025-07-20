"""
点云编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointEncoder(nn.Module):
    """
    点云编码器
    """
    
    def __init__(self, feat_dim: int, emb_dim: int):
        """
        初始化编码器
        
        Args:
            feat_dim: 输入特征维度
            emb_dim: 输出嵌入维度
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        
        # 点云特征编码网络
        self.point_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )
        
        # 全局池化后的处理网络
        self.global_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 3)  # 输出3D坐标
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入点云 (B, N, F)
        
        Returns:
            global_feat: 全局特征 (B, 3)
        """
        B, N, F = x.shape
        
        # 编码每个点的特征
        point_features = self.point_mlp(x)  # (B, N, emb_dim)
        
        # 全局平均池化
        global_feat = torch.mean(point_features, dim=1)  # (B, emb_dim)
        
        # 生成全局特征
        global_feat = self.global_mlp(global_feat)  # (B, 3)
        
        return global_feat 