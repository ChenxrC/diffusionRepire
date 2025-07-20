"""
血管树扩散训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import numpy as np

from ..core.noise_schedule import linear_beta_schedule
from ..dataset import TreeNormalDiffusionDataset
from ..models import CondNoisePredictor
from .monitoring import TrainingMonitor


class TreeDiffusionTrainer:
    """
    血管树扩散训练器
    """
    
    def __init__(self, 
                 model: CondNoisePredictor,
                 train_files: List[str],
                 device: str = 'cpu',
                 grid_size: int = 32,
                 point_spacing: float = 0.2,
                 batch_size: int = 2,
                 learning_rate: float = 1e-4,
                 T: int = 100):
        """
        初始化训练器
        
        Args:
            model: 噪声预测模型
            train_files: 训练文件列表
            device: 设备
            grid_size: 网格大小
            point_spacing: 点间距
            batch_size: 批次大小
            learning_rate: 学习率
            T: 时间步数
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.T = T
        
        # 创建数据集和数据加载器
        self.dataset = TreeNormalDiffusionDataset(train_files, grid_size, point_spacing)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 噪声调度
        self.betas = linear_beta_schedule(T)
        
        # 训练监控器
        self.monitor = TrainingMonitor()
        
        print(f"训练器初始化完成:")
        print(f"  - 数据集大小: {len(self.dataset)}")
        print(f"  - 批次大小: {batch_size}")
        print(f"  - 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Returns:
            avg_loss: 平均损失
            avg_noise_error: 平均噪声预测误差
        """
        self.model.train()
        total_loss = 0
        total_noise_error = 0
        num_batches = 0
        
        for feats, clean in self.dataloader:
            feats, clean = feats.to(self.device), clean.to(self.device)
            B = clean.shape[0]
            
            # 随机时间步
            t = torch.randint(0, self.T, (B,), device=self.device)
            beta_t = self.betas[t].unsqueeze(1)
            
            # 生成噪声
            noise = torch.randn_like(clean)
            noise_step = noise / self.T
            noisy = clean.clone()
            
            # 逐步加噪声
            for step in range(self.T):
                noisy = torch.sqrt(1-beta_t)*noisy + torch.sqrt(beta_t)*noise_step
            
            # 预测噪声
            pred_noise = self.model(feats, noisy, t)
            
            # 计算损失
            loss = nn.functional.mse_loss(pred_noise, noise)
            noise_error = nn.functional.mse_loss(pred_noise, noise)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_noise_error += noise_error.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_noise_error = total_noise_error / num_batches
        
        return avg_loss, avg_noise_error
    
    def train(self, epochs: int, save_interval: int = 100, 
              viz_interval: int = 500) -> dict:
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            save_interval: 保存间隔
            viz_interval: 可视化间隔
        
        Returns:
            training_history: 训练历史
        """
        print(f"开始训练 {epochs} 个epoch...")
        
        for epoch in range(epochs):
            # 训练一个epoch
            avg_loss, avg_noise_error = self.train_epoch()
            
            # 记录训练信息
            self.monitor.record_epoch(epoch, avg_loss, avg_noise_error)
            
            # 打印进度
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}, Noise Error: {avg_noise_error:.6f}")
                
                # 生成训练曲线
                self.monitor.plot_training_curves(epoch)
            
            # 可视化训练步骤
            if epoch % viz_interval == 0:
                self.monitor.visualize_training_step(self.dataloader, self.model, 
                                                   self.device, epoch, self.T, self.betas)
            
            # 保存模型
            if epoch % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
        
        print("训练完成!")
        return self.monitor.get_training_history()
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'betas': self.betas,
            'model_info': self.model.get_model_info()
        }
        torch.save(checkpoint, filename)
        print(f"检查点已保存: {filename}")
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.betas = checkpoint['betas']
        print(f"检查点已加载: {filename}")
    
    def validate_dataset(self):
        """验证数据集"""
        self.monitor.validate_dataset(self.dataset, self.device) 