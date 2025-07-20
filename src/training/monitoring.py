"""
训练监控模块
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader


class TrainingMonitor:
    """
    训练监控器
    """
    
    def __init__(self):
        """初始化监控器"""
        self.losses = []
        self.noise_errors = []
        self.data_quality_metrics = {}
        
        # 创建可视化目录
        os.makedirs("training_visualization", exist_ok=True)
    
    def record_epoch(self, epoch: int, loss: float, noise_error: float):
        """
        记录一个epoch的训练信息
        
        Args:
            epoch: 当前epoch
            loss: 损失值
            noise_error: 噪声预测误差
        """
        self.losses.append(loss)
        self.noise_errors.append(noise_error)
    
    def plot_training_curves(self, current_epoch: int):
        """
        绘制训练曲线
        
        Args:
            current_epoch: 当前epoch
        """
        if len(self.losses) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.losses, 'b-', alpha=0.7)
        axes[0, 0].set_title('训练损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 噪声误差曲线
        axes[0, 1].plot(self.noise_errors, 'r-', alpha=0.7)
        axes[0, 1].set_title('噪声预测误差')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Noise Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 损失对数曲线
        if len(self.losses) > 10:
            axes[1, 0].semilogy(self.losses, 'b-', alpha=0.7)
            axes[1, 0].set_title('训练损失 (对数)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss (log)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 最近100个epoch的损失
        if len(self.losses) > 100:
            recent_losses = self.losses[-100:]
            axes[1, 1].plot(recent_losses, 'g-', alpha=0.7)
            axes[1, 1].set_title('最近100个Epoch的损失')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"training_visualization/training_curves_epoch_{current_epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: {save_path}")
    
    def validate_dataset(self, dataset, device: str):
        """
        验证数据集
        
        Args:
            dataset: 数据集
            device: 设备
        """
        print("=== 数据集验证 ===")
        
        # 检查数据集大小
        print(f"数据集大小: {len(dataset)}")
        
        # 检查第一个样本
        if len(dataset) > 0:
            feats, target = dataset[0]
            print(f"特征形状: {feats.shape}")
            print(f"目标形状: {target.shape}")
            
            # 检查数据范围
            print(f"特征范围: [{feats.min():.3f}, {feats.max():.3f}]")
            print(f"目标范围: [{target.min():.3f}, {target.max():.3f}]")
            
            # 检查是否有NaN或无穷大
            if torch.isnan(feats).any() or torch.isinf(feats).any():
                print("⚠️  警告: 特征数据包含NaN或无穷大值")
            if torch.isnan(target).any() or torch.isinf(target).any():
                print("⚠️  警告: 目标数据包含NaN或无穷大值")
        
        print("数据集验证完成")
    
    def visualize_training_step(self, dataloader: DataLoader, model, device: str, 
                              epoch: int, T: int, betas: torch.Tensor):
        """
        可视化训练步骤
        
        Args:
            dataloader: 数据加载器
            model: 模型
            device: 设备
            epoch: 当前epoch
            T: 时间步数
            betas: beta序列
        """
        model.eval()
        
        # 获取一个批次的数据
        for feats, clean in dataloader:
            feats, clean = feats.to(device), clean.to(device)
            B = clean.shape[0]
            
            # 随机时间步
            t = torch.randint(0, T, (B,), device=device)
            beta_t = betas[t].unsqueeze(1)
            
            # 生成噪声
            noise = torch.randn_like(clean)
            noise_step = noise / T
            noisy = clean.clone()
            
            # 逐步加噪声
            for step in range(T):
                noisy = torch.sqrt(1-beta_t)*noisy + torch.sqrt(beta_t)*noise_step
            
            # 预测噪声
            with torch.no_grad():
                pred_noise = model(feats, noisy, t)
            
            # 可视化
            self._visualize_training_step_detail(feats, clean, noisy, pred_noise, noise, t, epoch, device)
            break  # 只处理第一个批次
    
    def _visualize_training_step_detail(self, feats, clean, noisy, pred_noise, true_noise, t, epoch, device):
        """
        详细可视化训练步骤
        """
        # 转换为numpy
        feats_np = feats.cpu().numpy()
        clean_np = clean.cpu().numpy()
        noisy_np = noisy.cpu().numpy()
        pred_noise_np = pred_noise.cpu().numpy()
        true_noise_np = true_noise.cpu().numpy()
        t_np = t.cpu().numpy()
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 原始点云
        axes[0, 0].scatter(feats_np[0, :, 0], feats_np[0, :, 1], alpha=0.6, s=1)
        axes[0, 0].set_title('原始点云 (XY投影)')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 目标曲面
        target_reshaped = clean_np[0].reshape(-1, 3)
        axes[0, 1].scatter(target_reshaped[:, 0], target_reshaped[:, 1], alpha=0.6, s=1)
        axes[0, 1].set_title('目标曲面 (XY投影)')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 噪声输入
        noisy_reshaped = noisy_np[0].reshape(-1, 3)
        axes[0, 2].scatter(noisy_reshaped[:, 0], noisy_reshaped[:, 1], alpha=0.6, s=1)
        axes[0, 2].set_title(f'噪声输入 (t={t_np[0]})')
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 预测噪声
        pred_noise_reshaped = pred_noise_np[0].reshape(-1, 3)
        axes[1, 0].scatter(pred_noise_reshaped[:, 0], pred_noise_reshaped[:, 1], alpha=0.6, s=1)
        axes[1, 0].set_title('预测噪声')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 真实噪声
        true_noise_reshaped = true_noise_np[0].reshape(-1, 3)
        axes[1, 1].scatter(true_noise_reshaped[:, 0], true_noise_reshaped[:, 1], alpha=0.6, s=1)
        axes[1, 1].set_title('真实噪声')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 噪声误差
        noise_error = pred_noise_reshaped - true_noise_reshaped
        axes[1, 2].scatter(noise_error[:, 0], noise_error[:, 1], alpha=0.6, s=1)
        axes[1, 2].set_title('噪声预测误差')
        axes[1, 2].set_xlabel('X')
        axes[1, 2].set_ylabel('Y')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"training_visualization/training_step_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"训练步骤可视化已保存: {save_path}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        获取训练历史
        
        Returns:
            training_history: 训练历史字典
        """
        return {
            'losses': self.losses,
            'noise_errors': self.noise_errors,
            'data_quality_metrics': self.data_quality_metrics
        } 