import json
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pdb
from visual import compute_tree_plane_normals, calculate_perpendicular_plane, find_max_points_branches, generate_noisy_normals
from tree_plane_predictor import tree_points_to_array, PointEncoder
from cascade_transform import apply_cascade_transform_to_branches
import os
# 创建临时dataset实例来生成曲面
class TempDataset:
    def _generate_surface_grid(self, centerline_points, main_direction, grid_size, point_spacing):
        """
        在以centerline_points为中轴的曲面上生成网格点
        
        Args:
            centerline_points: 中轴线上的点 (N, 3)
            main_direction: 中轴线主方向
            grid_size: 网格大小
            point_spacing: 点间距
        
        Returns:
            surface_points: 曲面上的网格点 (grid_size, grid_size, 3)
        """
        # 计算网格范围
        grid_extent = (grid_size - 1) * point_spacing / 2
        
        # 沿中轴线生成等距位置
        axis_positions = []
        for i in range(grid_size):
            t = i / (grid_size - 1)  # 0 到 1
            axis_pos = self._interpolate_on_centerline(centerline_points, t)
            axis_positions.append(axis_pos)
        
        axis_positions = np.array(axis_positions)
        surface_points = np.zeros((grid_size, grid_size, 3))
        
        # 为每个轴位置构建垂直平面的坐标系
        for i, axis_pos in enumerate(axis_positions):
            # 计算在该位置处中轴线的切线方向
            if i == 0:
                tangent = axis_positions[1] - axis_positions[0]
            elif i == grid_size - 1:
                tangent = axis_positions[-1] - axis_positions[-2]
            else:
                tangent = axis_positions[i+1] - axis_positions[i-1]
            
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            
            # 构建垂直于切线的两个正交基向量
            if abs(np.dot(tangent, np.array([1, 0, 0]))) < 0.9:
                base_vector = np.array([1, 0, 0])
            else:
                base_vector = np.array([0, 1, 0])
            
            u_axis = np.cross(tangent, base_vector)
            u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
            
            v_axis = np.cross(tangent, u_axis)
            v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-8)
            
            # 🔄 绕切线向量旋转90度
            u_axis_rotated = v_axis      # 原来的v_axis成为新的u_axis
            v_axis_rotated = -u_axis     # 原来的u_axis的负值成为新的v_axis
            
            # 在垂直平面上生成网格点
            for j in range(grid_size):
                # 从 -grid_extent 到 +grid_extent 均匀分布
                offset = (j / (grid_size - 1) - 0.5) * 2 * grid_extent
                
                # 使用旋转后的坐标系在垂直平面上的点
                point_on_surface = axis_pos + offset * u_axis_rotated
                
                # 添加轻微的曲率变化，使其更像真实的血管曲面
                curvature_factor = 0.1 * abs(offset) * np.sin(i * np.pi / grid_size)
                point_on_surface += curvature_factor * v_axis_rotated
                
                surface_points[i, j] = point_on_surface
        
        return surface_points
    
    def _interpolate_on_centerline(self, centerline_points, t):
        if len(centerline_points) == 1:
            return centerline_points[0]
        
        cumulative_lengths = [0]
        for i in range(1, len(centerline_points)):
            dist = np.linalg.norm(centerline_points[i] - centerline_points[i-1])
            cumulative_lengths.append(cumulative_lengths[-1] + dist)
        
        total_length = cumulative_lengths[-1]
        target_length = t * total_length
        
        for i in range(len(cumulative_lengths) - 1):
            if cumulative_lengths[i] <= target_length <= cumulative_lengths[i+1]:
                segment_t = (target_length - cumulative_lengths[i]) / (cumulative_lengths[i+1] - cumulative_lengths[i])
                return centerline_points[i] + segment_t * (centerline_points[i+1] - centerline_points[i])
        
        if t <= 0:
            return centerline_points[0]
        else:
            return centerline_points[-1]
# --------- Noise schedule utilities ---------

def linear_beta_schedule(T:int, beta_start:float=1e-4, beta_end:float=2e-2):
    return torch.linspace(beta_start, beta_end, T)

def generate_plane_points(center, normal, plane_size=30.0, grid_size=32):
    """在给定平面上生成均匀分布的点"""
    # 生成平面的两个正交基向量
    helper = np.array([1.,0.,0.])
    if np.allclose(abs(np.dot(helper, normal)),1.0,atol=1e-3):
        helper=np.array([0.,1.,0.])
    v1=np.cross(normal,helper); v1/=np.linalg.norm(v1)
    v2=np.cross(normal,v1); v2/=np.linalg.norm(v2)
    
    # 生成网格点
    g = np.linspace(-plane_size/2, plane_size/2, grid_size)
    u, v = np.meshgrid(g, g)
    
    # 计算平面上的点
    points = center + u[...,None]*v1 + v[...,None]*v2
    return points.reshape(-1, 3)  # (grid_size*grid_size, 3)


# --------- Dataset ---------
class TreeNormalDiffusionDataset(Dataset):
    def __init__(self, json_files: List[str], grid_size=32, point_spacing=0.2):
        self.files = json_files
        self.grid_size = grid_size
        self.point_spacing = point_spacing
        self.data = []
        self.targets = []
        for f in json_files:
            with open(f,'r') as fp:
                td=json.load(fp)
            # 生成增强数据
            augmented_trees = [td]
            for _ in range(3):  # 生成3个增强版本
                import copy
                td_aug = copy.deepcopy(td)
                td_aug["branches"] = apply_cascade_transform_to_branches(
                    td_aug["branches"],
                    max_offset=1.0,
                    rotation_range=30.0,
                    offset_strength=0.3,
                    rotation_strength=0.3
                )
                augmented_trees.append(td_aug)
            for td_aug in augmented_trees:
                pts = tree_points_to_array(td_aug)
                self.data.append(pts)
                trunk_pts, br1_pts, br2_pts = find_max_points_branches(td_aug)
                # 计算两个分支对应点的中点，构建中轴线
                min_len = min(len(br1_pts), len(br2_pts))
                if min_len > 0:
                    # 取相同数量的点来计算中点
                    br1_sampled = br1_pts[:min_len]
                    br2_sampled = br2_pts[:min_len]
                    midpoints = (br1_sampled + br2_sampled) / 2.0
                else:
                    # 如果没有分支点，使用主干末端作为中点
                    midpoints = np.array([trunk_pts[-1]])
                # 构建中轴线：主干点 + 分支中点
                centerline_points = np.vstack([trunk_pts, midpoints])
                # 对中轴线点进行排序，确保连续性
                centerline_center = centerline_points.mean(axis=0)
                centerline_centered = centerline_points - centerline_center
                # 使用PCA找到中轴线的主方向
                cov_matrix = np.cov(centerline_centered.T)
                eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
                idx = np.argsort(eigenvals)[::-1]
                main_direction = eigenvecs[:, idx[0]]  # 中轴线主方向
                # 将中轴线点投影到主方向上并排序
                projections = np.dot(centerline_centered, main_direction)
                sorted_indices = np.argsort(projections)
                sorted_centerline = centerline_points[sorted_indices]
                # 生成曲面上的32x32网格点
                surface_grid_points = self._generate_surface_grid(
                    sorted_centerline, main_direction, grid_size, point_spacing
                )
                self.targets.append(surface_grid_points.astype(np.float32))
    
    def _generate_surface_grid(self, centerline_points, main_direction, grid_size, point_spacing):
        """
        在以centerline_points为中轴的曲面上生成32x32网格点
        
        Args:
            centerline_points: 中轴线上的点 (N, 3)
            main_direction: 中轴线主方向
            grid_size: 网格大小 (32)
            point_spacing: 点间距
        
        Returns:
            surface_points: 曲面上的网格点 (grid_size, grid_size, 3)
        """
        # 计算网格范围
        grid_extent = (grid_size - 1) * point_spacing / 2
        
        # 生成沿中轴线方向的32个位置
        centerline_start = centerline_points[0]
        centerline_end = centerline_points[-1]
        centerline_length = np.linalg.norm(centerline_end - centerline_start)
        
        # 沿中轴线生成32个等距位置
        axis_positions = []
        for i in range(grid_size):
            t = i / (grid_size - 1)  # 0 到 1
            # 在中轴线上插值
            axis_pos = self._interpolate_on_centerline(centerline_points, t)
            axis_positions.append(axis_pos)
        
        axis_positions = np.array(axis_positions)
        
        # 为每个轴位置构建垂直平面的坐标系
        surface_points = np.zeros((grid_size, grid_size, 3))
        
        for i, axis_pos in enumerate(axis_positions):
            # 计算在该位置处中轴线的切线方向
            if i == 0:
                tangent = axis_positions[1] - axis_positions[0]
            elif i == grid_size - 1:
                tangent = axis_positions[-1] - axis_positions[-2]
            else:
                tangent = axis_positions[i+1] - axis_positions[i-1]
            
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            
            # 构建垂直于切线的两个正交基向量
            if abs(np.dot(tangent, np.array([1, 0, 0]))) < 0.9:
                base_vector = np.array([1, 0, 0])
            else:
                base_vector = np.array([0, 1, 0])
            
            u_axis = np.cross(tangent, base_vector)
            u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
            
            v_axis = np.cross(tangent, u_axis)
            v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-8)
            
            # 绕切线向量旋转90度：交换u_axis和v_axis，并对其中一个取负
            # 这相当于将坐标系绕tangent轴旋转90度
            u_axis_rotated = v_axis      # 原来的v_axis成为新的u_axis
            v_axis_rotated = -u_axis     # 原来的u_axis的负值成为新的v_axis
            
            # 在垂直平面上生成32个点（一行）
            for j in range(grid_size):
                # 从 -grid_extent 到 +grid_extent 均匀分布
                offset = (j / (grid_size - 1) - 0.5) * 2 * grid_extent
                
                # 使用旋转后的坐标系在垂直平面上的点
                point_on_surface = axis_pos + offset * u_axis_rotated
                
                # 添加轻微的曲率变化，使其更像真实的血管曲面
                curvature_factor = 0.1 * abs(offset) * np.sin(i * np.pi / grid_size)
                point_on_surface += curvature_factor * v_axis_rotated
                
                surface_points[i, j] = point_on_surface
        
        return surface_points
    
    def _interpolate_on_centerline(self, centerline_points, t):
        """
        在中轴线上按参数t插值 (t在0到1之间)
        """
        if len(centerline_points) == 1:
            return centerline_points[0]
        
        # 计算累积弧长
        cumulative_lengths = [0]
        for i in range(1, len(centerline_points)):
            dist = np.linalg.norm(centerline_points[i] - centerline_points[i-1])
            cumulative_lengths.append(cumulative_lengths[-1] + dist)
        
        total_length = cumulative_lengths[-1]
        target_length = t * total_length
        
        # 找到目标长度对应的线段
        for i in range(len(cumulative_lengths) - 1):
            if cumulative_lengths[i] <= target_length <= cumulative_lengths[i+1]:
                # 在该线段内插值
                segment_t = (target_length - cumulative_lengths[i]) / (cumulative_lengths[i+1] - cumulative_lengths[i])
                return centerline_points[i] + segment_t * (centerline_points[i+1] - centerline_points[i])
        
        # 边界情况
        if t <= 0:
            return centerline_points[0]
        else:
            return centerline_points[-1]
            
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        pts = self.data[idx]
        target = self.targets[idx]  # (grid_size, grid_size, 3)
        # normalize xyz
        xyz = pts[:,:3]
        xyz = xyz - xyz.mean(0, keepdims=True)
        xyz = xyz/(xyz.std()+1e-6)
        ids = pts[:,3:5]/100.0
        feat = np.concatenate([xyz, ids], axis=1)
        
        # 将目标展平为一维向量
        target_flat = target.flatten()  # (grid_size*grid_size*3,)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target_flat, dtype=torch.float32)

# --------- Conditioned Noise Predictor ---------
class CondNoisePredictor(nn.Module):
    def __init__(self, feat_dim=5, emb_dim=128, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.output_dim = grid_size * grid_size * 3  # 32*32*3 = 3072
        self.encoder = PointEncoder(feat_dim, emb_dim)
        self.time_fc = nn.Linear(1, emb_dim)
        self.cond_fc = nn.Linear(3, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + self.output_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, self.output_dim)
        )
    def forward(self, pts:torch.Tensor, noisy_points:torch.Tensor, t:torch.Tensor):
        # pts: (B,N,F)  noisy_points:(B,grid_size*grid_size*3) t:(B,)
        cond_raw = self.encoder(pts)           # (B,3)
        cond_emb = self.cond_fc(cond_raw)      # (B,emb_dim)
        time_emb = torch.sin(self.time_fc(t.float().unsqueeze(1)))  # (B,emb_dim)
        h = torch.cat([cond_emb + time_emb, noisy_points], dim=1)
        return self.mlp(h)

# --------- Training function ---------

def train_tree_diffusion(train_files: List[str], T:int=100, epochs:int=10000, batch_size:int=2, device='cpu', grid_size=32, point_spacing=0.2):
    # 首先验证数据集
    print("正在验证数据集...")
    dataset = TreeNormalDiffusionDataset(train_files, grid_size, point_spacing)
    _validate_dataset(dataset, device)
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CondNoisePredictor(feat_dim=5, emb_dim=128, grid_size=grid_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    betas = linear_beta_schedule(T)
    
    # 可视化相关变量
    viz_interval = 500  # 每500步可视化一次
    viz_counter = 0
    
    # 训练监控变量
    loss_history = []
    noise_error_history = []
    data_quality_metrics = []
    
    print(f"开始训练 - 数据集大小: {len(dataset)}, 批次大小: {batch_size}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for ep in range(epochs):
        total=0;cnt=0
        epoch_losses = []
        epoch_noise_errors = []
        
        for feats, clean in dl:
            feats, clean = feats.to(device), clean.to(device)
            B = clean.shape[0]
            t = torch.randint(0, T, (B,), device=device)
            beta_t = betas[t].unsqueeze(1)
            # 在循环之前生成随机噪声
            noise = torch.randn_like(clean)
            # 计算每一步的噪声增量
            noise_step = noise / T
            noisy = clean.clone()
            for step in range(T):
                # 按比例加入噪声
                noisy = torch.sqrt(1-beta_t)*noisy + torch.sqrt(beta_t)*noise_step
            pred_noise = model(feats, noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            
            # 记录损失和噪声预测误差
            total+=loss.item(); cnt+=1
            epoch_losses.append(loss.item())
            noise_error = F.mse_loss(pred_noise, noise).item()
            epoch_noise_errors.append(noise_error)
            
            # 单步可视化策略
            viz_counter += 1
            if viz_counter % viz_interval == 0:
                _visualize_training_step(feats, clean, noisy, pred_noise, noise, t, ep, viz_counter, device)
                
        # 记录epoch级别的统计信息
        avg_loss = total/cnt
        avg_noise_error = np.mean(epoch_noise_errors)
        loss_history.append(avg_loss)
        noise_error_history.append(avg_noise_error)
        
        # 计算数据质量指标
        if ep % 10 == 0:  # 每10个epoch计算一次
            quality_metrics = _calculate_data_quality_metrics(dataset, device)
            data_quality_metrics.append(quality_metrics)
        
        if ep%100==0:
            print(f"Epoch {ep}/{epochs} loss {avg_loss:.6f} noise_error {avg_noise_error:.6f}")
            
            # 绘制训练曲线
            if len(loss_history) > 1:
                _plot_training_curves(loss_history, noise_error_history, data_quality_metrics, ep)
    
    # 训练结束后的总结
    print("\n=== 训练完成总结 ===")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"最终噪声预测误差: {noise_error_history[-1]:.6f}")
    print(f"损失下降: {loss_history[0] - loss_history[-1]:.6f}")
    
    return model, betas

def _validate_dataset(dataset, device):
    """
    验证数据集的正确性
    """
    print("\n=== 数据集验证 ===")
    
    # 检查数据集大小
    print(f"数据集大小: {len(dataset)}")
    
    # 检查几个样本
    sample_indices = np.random.choice(len(dataset), min(3, len(dataset)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n样本 {i+1} (索引 {idx}):")
        
        feats, clean = dataset[idx]
        
        # 检查特征维度
        print(f"  特征形状: {feats.shape}")
        print(f"  目标形状: {clean.shape}")
        
        # 检查数值范围
        feats_np = feats.numpy()
        clean_np = clean.numpy()
        
        print(f"  特征范围: X[{feats_np[:, 0].min():.3f}, {feats_np[:, 0].max():.3f}], "
              f"Y[{feats_np[:, 1].min():.3f}, {feats_np[:, 1].max():.3f}], "
              f"Z[{feats_np[:, 2].min():.3f}, {feats_np[:, 2].max():.3f}]")
        
        grid_size = int(np.sqrt(clean.shape[0] // 3))
        clean_reshaped = clean_np.reshape(grid_size, grid_size, 3)
        print(f"  目标范围: X[{clean_reshaped[:, :, 0].min():.3f}, {clean_reshaped[:, :, 0].max():.3f}], "
              f"Y[{clean_reshaped[:, :, 1].min():.3f}, {clean_reshaped[:, :, 1].max():.3f}], "
              f"Z[{clean_reshaped[:, :, 2].min():.3f}, {clean_reshaped[:, :, 2].max():.3f}]")
        
        # 检查是否有NaN或无穷大值
        if np.any(np.isnan(feats_np)) or np.any(np.isnan(clean_np)):
            print("  ⚠️  警告: 发现NaN值!")
        if np.any(np.isinf(feats_np)) or np.any(np.isinf(clean_np)):
            print("  ⚠️  警告: 发现无穷大值!")
        
        # 检查曲面连续性
        dx = np.gradient(clean_reshaped, axis=0)
        dy = np.gradient(clean_reshaped, axis=1)
        grad_magnitude = np.sqrt(np.sum(dx**2, axis=2) + np.sum(dy**2, axis=2))
        max_grad = np.max(grad_magnitude)
        print(f"  最大梯度: {max_grad:.3f}")
        
        if max_grad > 10.0:
            print("  ⚠️  警告: 曲面梯度过大，可能存在不连续")
    
    # 可视化第一个样本
    print("\n正在生成验证可视化...")
    feats, clean = dataset[0]
    feats = feats.unsqueeze(0).to(device)
    clean = clean.unsqueeze(0).to(device)
    
    _visualize_validation_sample(feats, clean, device)
    
    print("✅ 数据集验证完成")

def _visualize_validation_sample(feats, clean, device):
    """
    可视化验证样本
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 提取数据
    feats_np = feats[0].cpu().numpy()
    original_points = feats_np[:, :3]
    
    grid_size = int(np.sqrt(clean.shape[1] // 3))
    clean_np = clean[0].cpu().numpy().reshape(grid_size, grid_size, 3)
    
    # 创建可视化
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('数据集验证可视化', fontsize=16)
    
    # 原始点云
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
               c='blue', s=1, alpha=0.6)
    ax1.set_title('原始血管点云')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # 目标曲面
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(clean_np[:, :, 0], clean_np[:, :, 1], clean_np[:, :, 2], 
                    alpha=0.7, cmap='viridis')
    ax2.set_title('目标曲面')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # 点云与曲面对比
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
               c='blue', s=1, alpha=0.6, label='原始点云')
    ax3.plot_surface(clean_np[:, :, 0], clean_np[:, :, 1], clean_np[:, :, 2], 
                    alpha=0.3, cmap='viridis', label='目标曲面')
    ax3.set_title('点云与曲面对比')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = "training_visualization"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dataset_validation.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"验证可视化已保存: {save_path}")

def _calculate_data_quality_metrics(dataset, device):
    """
    计算数据集质量指标
    """
    metrics = {}
    
    # 随机采样几个样本进行分析
    sample_indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    
    surface_areas = []
    surface_smoothness = []
    point_cloud_coverage = []
    
    for idx in sample_indices:
        feats, clean = dataset[idx]
        feats = feats.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)
        
        # 计算曲面面积
        grid_size = int(np.sqrt(clean.shape[1] // 3))
        clean_reshaped = clean[0].cpu().numpy().reshape(grid_size, grid_size, 3)
        
        # 计算曲面面积（简化版本）
        dx = np.gradient(clean_reshaped, axis=0)
        dy = np.gradient(clean_reshaped, axis=1)
        area_elements = np.sqrt(np.sum(dx**2, axis=2) + np.sum(dy**2, axis=2))
        surface_area = np.sum(area_elements)
        surface_areas.append(surface_area)
        
        # 计算曲面平滑度
        laplacian = np.gradient(np.gradient(clean_reshaped, axis=0), axis=0) + \
                   np.gradient(np.gradient(clean_reshaped, axis=1), axis=1)
        smoothness = np.mean(np.linalg.norm(laplacian, axis=2))
        surface_smoothness.append(smoothness)
        
        # 计算点云覆盖度
        original_points = feats[0, :, :3].cpu().numpy()
        surface_center = np.mean(clean_reshaped, axis=(0, 1))
        distances = np.linalg.norm(original_points - surface_center, axis=1)
        coverage = np.mean(distances < np.std(distances) * 2)
        point_cloud_coverage.append(coverage)
    
    metrics['avg_surface_area'] = np.mean(surface_areas)
    metrics['avg_smoothness'] = np.mean(surface_smoothness)
    metrics['avg_coverage'] = np.mean(point_cloud_coverage)
    
    return metrics

def _plot_training_curves(loss_history, noise_error_history, data_quality_metrics, current_epoch):
    """
    绘制训练曲线
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'训练监控 - Epoch {current_epoch}', fontsize=16)
    
    # 损失曲线
    axes[0, 0].plot(loss_history, 'b-', label='总损失')
    axes[0, 0].set_title('训练损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 噪声预测误差
    axes[0, 1].plot(noise_error_history, 'r-', label='噪声预测误差')
    axes[0, 1].set_title('噪声预测误差')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 数据质量指标
    if data_quality_metrics:
        epochs_quality = list(range(0, len(data_quality_metrics) * 10, 10))
        surface_areas = [m['avg_surface_area'] for m in data_quality_metrics]
        smoothness = [m['avg_smoothness'] for m in data_quality_metrics]
        
        axes[1, 0].plot(epochs_quality, surface_areas, 'g-', label='平均曲面面积')
        axes[1, 0].set_title('数据质量 - 曲面面积')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('面积')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(epochs_quality, smoothness, 'm-', label='平均平滑度')
        axes[1, 1].set_title('数据质量 - 曲面平滑度')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('平滑度')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = "training_visualization"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"training_curves_{current_epoch}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存: {save_path}")

def _visualize_training_step(feats, clean, noisy, pred_noise, true_noise, t, epoch, step, device):
    """
    可视化训练过程中的单步数据，用于监测训练正确性
    
    Args:
        feats: 输入特征 (B, N, F)
        clean: 干净的目标曲面 (B, grid_size*grid_size*3)
        noisy: 加噪声的曲面 (B, grid_size*grid_size*3)
        pred_noise: 预测的噪声 (B, grid_size*grid_size*3)
        true_noise: 真实的噪声 (B, grid_size*grid_size*3)
        t: 时间步 (B,)
        epoch: 当前epoch
        step: 当前步数
        device: 设备
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 只可视化第一个样本
    batch_idx = 0
    grid_size = int(np.sqrt(clean.shape[1] // 3))
    
    # 将张量转换为numpy数组并重塑
    clean_np = clean[batch_idx].cpu().detach().numpy().reshape(grid_size, grid_size, 3)
    noisy_np = noisy[batch_idx].cpu().detach().numpy().reshape(grid_size, grid_size, 3)
    pred_noise_np = pred_noise[batch_idx].cpu().detach().numpy().reshape(grid_size, grid_size, 3)
    true_noise_np = true_noise[batch_idx].cpu().detach().numpy().reshape(grid_size, grid_size, 3)
    
    # 提取原始点云数据
    feats_np = feats[batch_idx].cpu().detach().numpy()
    original_points = feats_np[:, :3]  # 只取xyz坐标
    
    # 创建子图
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'训练步骤可视化 - Epoch {epoch}, Step {step}, Time Step {t[batch_idx].item()}', fontsize=16)
    
    # 1. 原始点云
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
               c='blue', s=1, alpha=0.6, label='原始点云')
    ax1.set_title('原始血管点云')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()
    
    # 2. 目标曲面 (clean)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(clean_np[:, :, 0], clean_np[:, :, 1], clean_np[:, :, 2], 
                    alpha=0.7, cmap='viridis')
    ax2.set_title('目标曲面 (Label)')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # 3. 加噪声的曲面
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.plot_surface(noisy_np[:, :, 0], noisy_np[:, :, 1], noisy_np[:, :, 2], 
                    alpha=0.7, cmap='plasma')
    ax3.set_title('加噪声曲面 (Input)')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    
    # 4. 真实噪声
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot_surface(true_noise_np[:, :, 0], true_noise_np[:, :, 1], true_noise_np[:, :, 2], 
                    alpha=0.7, cmap='coolwarm')
    ax4.set_title('真实噪声 (Ground Truth)')
    ax4.set_xlabel('X'); ax4.set_ylabel('Y'); ax4.set_zlabel('Z')
    
    # 5. 预测噪声
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.plot_surface(pred_noise_np[:, :, 0], pred_noise_np[:, :, 1], pred_noise_np[:, :, 2], 
                    alpha=0.7, cmap='coolwarm')
    ax5.set_title('预测噪声 (Prediction)')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y'); ax5.set_zlabel('Z')
    
    # 6. 噪声差异 (预测 - 真实)
    noise_diff = pred_noise_np - true_noise_np
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    im = ax6.plot_surface(noise_diff[:, :, 0], noise_diff[:, :, 1], noise_diff[:, :, 2], 
                         alpha=0.7, cmap='RdBu')
    ax6.set_title('噪声预测误差')
    ax6.set_xlabel('X'); ax6.set_ylabel('Y'); ax6.set_zlabel('Z')
    
    # 添加颜色条
    fig.colorbar(im, ax=ax6, shrink=0.5, aspect=5)
    
    # 计算和显示统计信息
    noise_mse = np.mean((pred_noise_np - true_noise_np) ** 2)
    noise_mae = np.mean(np.abs(pred_noise_np - true_noise_np))
    
    # 在图上添加统计信息
    stats_text = f'噪声MSE: {noise_mse:.6f}\n噪声MAE: {noise_mae:.6f}'
    fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = "training_visualization"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"training_step_{epoch}_{step}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"训练可视化已保存: {save_path}")
    print(f"噪声预测误差 - MSE: {noise_mse:.6f}, MAE: {noise_mae:.6f}")
    
    # 检查数据合理性
    _check_training_data_validity(clean_np, noisy_np, pred_noise_np, true_noise_np, original_points)
    
    plt.close()

def _check_training_data_validity(clean, noisy, pred_noise, true_noise, original_points):
    """
    检查训练数据的合理性
    
    Args:
        clean: 目标曲面
        noisy: 加噪声曲面
        pred_noise: 预测噪声
        true_noise: 真实噪声
        original_points: 原始点云
    """
    print("\n=== 训练数据合理性检查 ===")
    
    # 1. 检查曲面形状
    clean_range = np.ptp(clean, axis=(0, 1))  # 每个维度的范围
    noisy_range = np.ptp(noisy, axis=(0, 1))
    print(f"目标曲面范围: X[{clean_range[0]:.3f}], Y[{clean_range[1]:.3f}], Z[{clean_range[2]:.3f}]")
    print(f"噪声曲面范围: X[{noisy_range[0]:.3f}], Y[{noisy_range[1]:.3f}], Z[{noisy_range[2]:.3f}]")
    
    # 2. 检查噪声幅度
    noise_magnitude = np.linalg.norm(true_noise, axis=2)
    pred_magnitude = np.linalg.norm(pred_noise, axis=2)
    print(f"真实噪声幅度: 均值={np.mean(noise_magnitude):.3f}, 标准差={np.std(noise_magnitude):.3f}")
    print(f"预测噪声幅度: 均值={np.mean(pred_magnitude):.3f}, 标准差={np.std(pred_magnitude):.3f}")
    
    # 3. 检查点云与曲面的关系
    original_center = np.mean(original_points, axis=0)
    surface_center = np.mean(clean, axis=(0, 1))
    distance = np.linalg.norm(original_center - surface_center)
    print(f"原始点云中心与曲面中心距离: {distance:.3f}")
    
    # 4. 检查曲面连续性
    clean_grad_x = np.gradient(clean, axis=0)
    clean_grad_y = np.gradient(clean, axis=1)
    grad_magnitude = np.sqrt(np.sum(clean_grad_x**2 + clean_grad_y**2, axis=2))
    print(f"曲面梯度幅度: 均值={np.mean(grad_magnitude):.3f}, 最大值={np.max(grad_magnitude):.3f}")
    
    # 5. 检查是否有异常值
    clean_std = np.std(clean, axis=(0, 1))
    noisy_std = np.std(noisy, axis=(0, 1))
    print(f"目标曲面标准差: X[{clean_std[0]:.3f}], Y[{clean_std[1]:.3f}], Z[{clean_std[2]:.3f}]")
    print(f"噪声曲面标准差: X[{noisy_std[0]:.3f}], Y[{noisy_std[1]:.3f}], Z[{noisy_std[2]:.3f}]")
    
    # 6. 合理性判断
    issues = []
    if distance > 5.0:
        issues.append("原始点云与曲面中心距离过大")
    if np.max(grad_magnitude) > 10.0:
        issues.append("曲面梯度过大，可能存在不连续")
    if np.any(clean_std > 5.0):
        issues.append("目标曲面标准差过大")
    if np.mean(noise_magnitude) > 2.0:
        issues.append("噪声幅度过大")
    
    if issues:
        print("⚠️  发现潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ 数据看起来合理")
    
    print("=" * 40)

# --------- Denoise ---------

def denoise_with_tree(tree_json:str, model:CondNoisePredictor, betas:torch.Tensor, device='cpu', grid_size=32, point_spacing=0.2):
    with open(tree_json,'r') as fp:
        td=json.load(fp)
    pts_arr = tree_points_to_array(td)
    xyz = pts_arr[:,:3]
    xyz = xyz-xyz.mean(0,keepdims=True)
    xyz = xyz/(xyz.std()+1e-6)
    ids = pts_arr[:,3:5]/100.0
    feats = torch.tensor(np.concatenate([xyz,ids],axis=1)[None,...],dtype=torch.float32,device=device)
    T = betas.shape[0]
    
    # 获取主干点和分支点
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
    all_branch_pts = np.vstack([br1_pts, br2_pts])
    
    # 1. 计算所有点的中心作为平面中心
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    plane_center = all_points.mean(axis=0)
    
    # 2. 找到经过分支最多的平面（分支点的主平面）
    branch_center = all_branch_pts.mean(axis=0)
    branch_centered = all_branch_pts - branch_center
    
    # 计算分支点的主平面法向量
    if len(branch_centered) >= 3:
        branch_cov = np.cov(branch_centered.T)
        branch_eigenvals, branch_eigenvecs = np.linalg.eigh(branch_cov)
        branch_idx = np.argsort(branch_eigenvals)[::-1]
        
        # 分支主平面由前两个主成分确定，法向量是第三个主成分
        branch_plane_normal = branch_eigenvecs[:, branch_idx[2]]  # 最小特征值对应方向
    else:
        # 如果分支点太少，使用默认法向量
        branch_plane_normal = np.array([0, 0, 1])
    
    # 确保法向量是单位向量
    branch_plane_normal = branch_plane_normal / np.linalg.norm(branch_plane_normal)
    
    # 3. 找到与分支平面垂直且经过主干最多的平面
    trunk_center = trunk_pts.mean(axis=0)
    trunk_centered = trunk_pts - trunk_center
    
    # 将主干点投影到垂直于分支平面法向量的空间中
    projection_matrix = np.eye(3) - np.outer(branch_plane_normal, branch_plane_normal)
    trunk_projected = trunk_centered @ projection_matrix.T
    
    # 在投影空间中找主干点的主方向
    if len(trunk_projected) >= 2:
        trunk_proj_cov = np.cov(trunk_projected.T)
        trunk_eigenvals, trunk_eigenvecs = np.linalg.eigh(trunk_proj_cov)
        trunk_idx = np.argsort(trunk_eigenvals)[::-1]
        
        # 主干在投影空间中的主方向
        trunk_main_dir_projected = trunk_eigenvecs[:, trunk_idx[0]]
    else:
        trunk_main_dir_projected = np.array([1, 0, 0])
        trunk_main_dir_projected = trunk_main_dir_projected - np.dot(trunk_main_dir_projected, branch_plane_normal) * branch_plane_normal
    
    # 确保主干主方向是单位向量
    trunk_main_dir_projected = trunk_main_dir_projected / (np.linalg.norm(trunk_main_dir_projected) + 1e-8)
    
    # 4. 构建初始平面的坐标系
    plane_normal = branch_plane_normal
    u_axis = trunk_main_dir_projected
    v_axis = np.cross(plane_normal, u_axis)
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-8)
    u_axis = np.cross(v_axis, plane_normal)
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
    
    # 5. 在平面上生成32x32的正方形网格，间隔为point_spacing
    grid_extent = (grid_size - 1) * point_spacing / 2
    u_coords = np.linspace(-grid_extent, grid_extent, grid_size)
    v_coords = np.linspace(-grid_extent, grid_extent, grid_size)
    U, V = np.meshgrid(u_coords, v_coords)
    
    # 将网格点转换到3D空间
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            point_3d = plane_center + U[i,j] * u_axis + V[i,j] * v_axis
            grid_points.append(point_3d)
    
    # 转换为torch tensor
    initial_points = np.array(grid_points).flatten()
    x = torch.tensor(initial_points, dtype=torch.float32, device=device).unsqueeze(0)
    
    print(f"初始平面信息:")
    print(f"  平面中心: {plane_center}")
    print(f"  分支平面法向量: {branch_plane_normal}")
    print(f"  主干主方向: {trunk_main_dir_projected}")
    print(f"  平面法向量: {plane_normal}")
    print(f"  网格范围: {grid_extent*2:.2f} x {grid_extent*2:.2f}, 点间距: {point_spacing}")
    
    with torch.no_grad():
        for t_inv in range(T-1,-1,-1):
            t=torch.tensor([t_inv],device=device)
            beta=betas[t_inv]
            pred_noise = model(feats, x, t)
            x = (x - torch.sqrt(beta)*pred_noise)/torch.sqrt(1-beta)
    return x.squeeze().cpu().numpy().reshape(grid_size, grid_size, 3)

# --------- Denoise with GIF ---------

def denoise_with_gif(tree_json:str, model:CondNoisePredictor, betas:torch.Tensor, gif_path:str='denoise.gif', device='cpu', grid_size=32, point_spacing=0.2):
    """可视化32*32个点在去噪过程中的移动"""
    import imageio, matplotlib.pyplot as plt
    with open(tree_json,'r') as fp:
        td=json.load(fp)
    pts_arr = tree_points_to_array(td)
    xyz = pts_arr[:,:3]
    center = xyz.mean(axis=0)
    # normalize feats same as before
    xyz_n = xyz-center
    xyz_n = xyz_n/(xyz_n.std()+1e-6)
    ids = pts_arr[:,3:5]/100.0
    feats = torch.tensor(np.concatenate([xyz_n,ids],axis=1)[None,...],dtype=torch.float32,device=device)
    T = betas.shape[0]
    
    # 获取主干点和分支点
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
    all_branch_pts = np.vstack([br1_pts, br2_pts])
    
    # 1. 计算所有点的中心作为平面中心
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    plane_center = all_points.mean(axis=0)
    
    # 2. 找到经过分支最多的平面（分支点的主平面）
    branch_center = all_branch_pts.mean(axis=0)
    branch_centered = all_branch_pts - branch_center
    
    # 计算分支点的主平面法向量
    if len(branch_centered) >= 3:
        branch_cov = np.cov(branch_centered.T)
        branch_eigenvals, branch_eigenvecs = np.linalg.eigh(branch_cov)
        branch_idx = np.argsort(branch_eigenvals)[::-1]
        
        # 分支主平面由前两个主成分确定，法向量是第三个主成分
        branch_plane_normal = branch_eigenvecs[:, branch_idx[2]]  # 最小特征值对应方向
    else:
        # 如果分支点太少，使用默认法向量
        branch_plane_normal = np.array([0, 0, 1])
    
    # 确保法向量是单位向量
    branch_plane_normal = branch_plane_normal / np.linalg.norm(branch_plane_normal)
    
    # 3. 找到与分支平面垂直且经过主干最多的平面
    trunk_center = trunk_pts.mean(axis=0)
    trunk_centered = trunk_pts - trunk_center
    
    # 将主干点投影到垂直于分支平面法向量的空间中
    projection_matrix = np.eye(3) - np.outer(branch_plane_normal, branch_plane_normal)
    trunk_projected = trunk_centered @ projection_matrix.T
    
    # 在投影空间中找主干点的主方向
    if len(trunk_projected) >= 2:
        trunk_proj_cov = np.cov(trunk_projected.T)
        trunk_eigenvals, trunk_eigenvecs = np.linalg.eigh(trunk_proj_cov)
        trunk_idx = np.argsort(trunk_eigenvals)[::-1]
        
        # 主干在投影空间中的主方向
        trunk_main_dir_projected = trunk_eigenvecs[:, trunk_idx[0]]
    else:
        trunk_main_dir_projected = np.array([1, 0, 0])
        trunk_main_dir_projected = trunk_main_dir_projected - np.dot(trunk_main_dir_projected, branch_plane_normal) * branch_plane_normal
    
    # 确保主干主方向是单位向量
    trunk_main_dir_projected = trunk_main_dir_projected / (np.linalg.norm(trunk_main_dir_projected) + 1e-8)
    
    # 4. 构建初始平面的坐标系
    plane_normal = branch_plane_normal
    u_axis = trunk_main_dir_projected
    v_axis = np.cross(plane_normal, u_axis)
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-8)
    u_axis = np.cross(v_axis, plane_normal)
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
    
    # 5. 在平面上生成32x32的正方形网格，间隔为point_spacing
    grid_extent = (grid_size - 1) * point_spacing / 2
    u_coords = np.linspace(-grid_extent, grid_extent, grid_size)
    v_coords = np.linspace(-grid_extent, grid_extent, grid_size)
    U, V = np.meshgrid(u_coords, v_coords)
    
    # 将网格点转换到3D空间
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            point_3d = plane_center + U[i,j] * u_axis + V[i,j] * v_axis
            grid_points.append(point_3d)
    
    # 转换为torch tensor
    initial_points = np.array(grid_points).flatten()
    x = torch.tensor(initial_points, dtype=torch.float32, device=device).unsqueeze(0)
    
    print(f"GIF模式 - 初始平面信息:")
    print(f"  平面中心: {plane_center}")
    print(f"  分支平面法向量: {branch_plane_normal}")
    print(f"  主干主方向: {trunk_main_dir_projected}")
    print(f"  网格范围: {grid_extent*2:.2f} x {grid_extent*2:.2f}, 点间距: {point_spacing}")
    
    frames=[]
    # precompute point sets for scatter
    branch_pts = np.vstack([br1_pts, br2_pts])

    with torch.no_grad():
        for t_inv in range(T-1,-1,-1):
            # 将当前点云重塑为(grid_size, grid_size, 3)格式
            current_points = x.squeeze().cpu().numpy().reshape(grid_size, grid_size, 3)
            
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制原始血管点
            ax.scatter(*trunk_pts.T,  c='blue',  s=2, alpha=0.6, label='主干')
            ax.scatter(*br1_pts.T,    c='green', s=2, alpha=0.6, label='分支1')
            ax.scatter(*br2_pts.T,    c='red',   s=2, alpha=0.6, label='分支2')
            
            # 绘制初始平面（半透明）
            plane_size = grid_extent * 1.5
            plane_u = np.linspace(-plane_size, plane_size, 10)
            plane_v = np.linspace(-plane_size, plane_size, 10)
            Plane_U, Plane_V = np.meshgrid(plane_u, plane_v)
            
            plane_points = np.zeros((10, 10, 3))
            for i in range(10):
                for j in range(10):
                    plane_points[i, j] = plane_center + Plane_U[i,j] * u_axis + Plane_V[i,j] * v_axis
            
            ax.plot_surface(plane_points[:,:,0], plane_points[:,:,1], plane_points[:,:,2], 
                          alpha=0.2, color='yellow', label='初始平面')
            
            # 绘制分支主平面（用于对比）
            branch_plane_points = np.zeros((10, 10, 3))
            # 构建分支平面的坐标系
            if abs(np.dot(branch_plane_normal, np.array([1, 0, 0]))) < 0.9:
                branch_u = np.cross(branch_plane_normal, np.array([1, 0, 0]))
            else:
                branch_u = np.cross(branch_plane_normal, np.array([0, 1, 0]))
            branch_u = branch_u / np.linalg.norm(branch_u)
            branch_v = np.cross(branch_plane_normal, branch_u)
            branch_v = branch_v / np.linalg.norm(branch_v)
            
            for i in range(10):
                for j in range(10):
                    branch_plane_points[i, j] = branch_center + Plane_U[i,j] * branch_u + Plane_V[i,j] * branch_v
            
            ax.plot_surface(branch_plane_points[:,:,0], branch_plane_points[:,:,1], branch_plane_points[:,:,2], 
                          alpha=0.1, color='cyan', label='分支主平面')
            
            # 绘制当前预测的点云
            points_flat = current_points.reshape(-1, 3)
            ax.scatter(*points_flat.T, c='orange', s=3, alpha=0.8, label='扩散点云')
            
            # 绘制点云的网格线以显示结构
            for i in range(0, grid_size, 4):  # 每4行绘制一条线
                line_points = current_points[i, :, :]
                ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                       color='red', alpha=0.5, linewidth=1)
            for j in range(0, grid_size, 4):  # 每4列绘制一条线
                line_points = current_points[:, j, :]
                ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                       color='red', alpha=0.5, linewidth=1)

            # 设置视图
            all_points_view = np.vstack([trunk_pts, br1_pts, br2_pts, points_flat])
            center_view = all_points_view.mean(axis=0)
            range_val = 40
            ax.set_xlim(center_view[0]-range_val, center_view[0]+range_val)
            ax.set_ylim(center_view[1]-range_val, center_view[1]+range_val)
            ax.set_zlim(center_view[2]-range_val, center_view[2]+range_val)
            ax.set_title(f'去噪步骤: {T-t_inv-1}/{T}\n初始平面：经过主干最多，与分支平面垂直\n网格: {grid_size}x{grid_size}, 间距: {point_spacing}')
            ax.legend()
            
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            frames.append(frame)
            plt.close(fig)

            # diffusion step
            if t_inv > 0:  # 避免最后一步
                t=torch.tensor([t_inv],device=device)
                beta=betas[t_inv]
                pred_noise = model(feats, x, t)
                x = (x - torch.sqrt(beta)*pred_noise)/torch.sqrt(1-beta)
                
    imageio.mimsave(gif_path, frames, fps=5)
    return x.squeeze().cpu().numpy().reshape(grid_size, grid_size, 3)

# --------- Visualization functions ---------

def safe_find_max_points_branches(tree_data):
    """安全的分支点提取函数，处理不同的数据结构"""
    try:
        # 尝试使用原始函数
        from visual import find_max_points_branches
        return find_max_points_branches(tree_data)
    except (KeyError, TypeError, IndexError) as e:
        print(f"原始函数失败: {e}")
        print("使用备用方法提取分支点...")
        
        # 备用方法：直接从branches列表中提取
        if "branches" in tree_data and isinstance(tree_data["branches"], list):
            if len(tree_data["branches"]) > 0:
                trunk_branch = tree_data["branches"][0]
                trunk_points = np.array(trunk_branch["points"], dtype=np.float32)
                
                # 获取分支
                children = trunk_branch.get("children", [])
                if len(children) >= 2:
                    # 按点数排序，取最大的两个分支
                    children_sorted = sorted(children, key=lambda b: len(b["points"]), reverse=True)
                    branch1_pts = np.array(children_sorted[0]["points"], dtype=np.float32)
                    branch2_pts = np.array(children_sorted[1]["points"], dtype=np.float32)
                elif len(children) == 1:
                    branch1_pts = np.array(children[0]["points"], dtype=np.float32)
                    branch2_pts = branch1_pts.copy()
                else:
                    # 没有分支，使用主干的前半部分和后半部分
                    mid_idx = len(trunk_points) // 2
                    branch1_pts = trunk_points[:mid_idx]
                    branch2_pts = trunk_points[mid_idx:]
                
                return trunk_points, branch1_pts, branch2_pts
        
        # 如果所有方法都失败，创建默认数据
        print("所有方法都失败，创建默认分支点...")
        default_points = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 2]], dtype=np.float32)
        return default_points, default_points, default_points

def visualize_generated_surface(tree_json: str, generated_points: np.ndarray, save_path: str = None, show_wireframe: bool = True, interactive: bool = True):
    """
    可视化生成的曲面与原始血管树的对比
    
    Args:
        tree_json: 血管树json文件路径
        generated_points: 生成的曲面点，形状为(grid_size, grid_size, 3)
        save_path: 保存图片的路径，如果为None则不保存
        show_wireframe: 是否显示网格线
        interactive: 是否显示交互式3D界面
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 读取原始血管数据
    with open(tree_json, 'r') as fp:
        td = json.load(fp)
    
    trunk_pts, br1_pts, br2_pts = safe_find_max_points_branches(td)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1: 原始血管树
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(*trunk_pts.T, c='blue', s=3, alpha=0.8, label='主干')
    ax1.scatter(*br1_pts.T, c='green', s=3, alpha=0.8, label='分支1')
    ax1.scatter(*br2_pts.T, c='red', s=3, alpha=0.8, label='分支2')
    ax1.set_title('原始血管树')
    ax1.legend()
    ax1.set_axis_off()
    
    # 子图2: 生成的曲面
    ax2 = fig.add_subplot(132, projection='3d')
    grid_size = generated_points.shape[0]
    
    # 绘制曲面
    X, Y, Z = generated_points[:, :, 0], generated_points[:, :, 1], generated_points[:, :, 2]
    surf = ax2.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', linewidth=0.1)
    
    if show_wireframe:
        # 添加网格线
        for i in range(0, grid_size, 4):
            ax2.plot(X[i, :], Y[i, :], Z[i, :], 'k-', alpha=0.3, linewidth=0.5)
        for j in range(0, grid_size, 4):
            ax2.plot(X[:, j], Y[:, j], Z[:, j], 'k-', alpha=0.3, linewidth=0.5)
    
    ax2.set_title('生成的曲面')
    ax2.set_axis_off()
    
    # 子图3: 叠加显示
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 绘制原始血管点
    ax3.scatter(*trunk_pts.T, c='blue', s=2, alpha=0.6, label='主干')
    ax3.scatter(*br1_pts.T, c='green', s=2, alpha=0.6, label='分支1')
    ax3.scatter(*br2_pts.T, c='red', s=2, alpha=0.6, label='分支2')
    
    # 绘制生成的曲面（半透明）
    ax3.plot_surface(X, Y, Z, alpha=0.3, color='orange')
    
    # 绘制曲面边界线
    ax3.plot(X[0, :], Y[0, :], Z[0, :], 'orange', linewidth=2, alpha=0.8)
    ax3.plot(X[-1, :], Y[-1, :], Z[-1, :], 'orange', linewidth=2, alpha=0.8)
    ax3.plot(X[:, 0], Y[:, 0], Z[:, 0], 'orange', linewidth=2, alpha=0.8)
    ax3.plot(X[:, -1], Y[:, -1], Z[:, -1], 'orange', linewidth=2, alpha=0.8)
    
    ax3.set_title('叠加对比')
    ax3.legend()
    ax3.set_axis_off()
    
    # 统一设置视图范围
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    center = all_points.mean(axis=0)
    range_val = 30
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
    
    plt.tight_layout()
    
    # 保存图片（如果需要）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    # 显示交互式界面（如果需要）
    if interactive:
        print("正在打开交互式3D可视化界面...")
        print("您可以:")
        print("- 拖拽旋转视角")
        print("- 滚轮缩放")
        print("- 关闭窗口继续程序执行")
        plt.show()
    else:
        plt.close()

def analyze_surface_quality(tree_json: str, generated_points: np.ndarray):
    """
    分析生成曲面的质量指标
    
    Args:
        tree_json: 血管树json文件路径
        generated_points: 生成的曲面点，形状为(grid_size, grid_size, 3)
    
    Returns:
        dict: 包含各种质量指标的字典
    """
    
    # 读取原始血管数据
    with open(tree_json, 'r') as fp:
        td = json.load(fp)
    
    trunk_pts, br1_pts, br2_pts = safe_find_max_points_branches(td)
    all_vessel_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    
    # 将生成的曲面点展平
    surface_points = generated_points.reshape(-1, 3)
    
    # 计算质量指标
    metrics = {}
    
    # 1. 曲面覆盖范围与血管范围的比较
    vessel_bbox = {
        'min': all_vessel_points.min(axis=0),
        'max': all_vessel_points.max(axis=0),
        'range': all_vessel_points.max(axis=0) - all_vessel_points.min(axis=0)
    }
    
    surface_bbox = {
        'min': surface_points.min(axis=0),
        'max': surface_points.max(axis=0),
        'range': surface_points.max(axis=0) - surface_points.min(axis=0)
    }
    
    metrics['vessel_bbox'] = vessel_bbox
    metrics['surface_bbox'] = surface_bbox
    metrics['coverage_ratio'] = surface_bbox['range'] / vessel_bbox['range']
    
    # 2. 曲面中心与血管中心的距离
    vessel_center = all_vessel_points.mean(axis=0)
    surface_center = surface_points.mean(axis=0)
    metrics['center_distance'] = np.linalg.norm(surface_center - vessel_center)
    
    # 3. 血管点到曲面的最小距离分布
    from scipy.spatial.distance import cdist
    distances = cdist(all_vessel_points, surface_points)
    min_distances = distances.min(axis=1)
    
    metrics['min_distance_stats'] = {
        'mean': min_distances.mean(),
        'std': min_distances.std(),
        'median': np.median(min_distances),
        'max': min_distances.max(),
        'min': min_distances.min()
    }
    
    # 4. 曲面平滑度（相邻点的距离变化）
    grid_size = generated_points.shape[0]
    
    # 水平方向的平滑度
    h_smoothness = []
    for i in range(grid_size):
        for j in range(grid_size - 1):
            dist = np.linalg.norm(generated_points[i, j+1] - generated_points[i, j])
            h_smoothness.append(dist)
    
    # 垂直方向的平滑度
    v_smoothness = []
    for i in range(grid_size - 1):
        for j in range(grid_size):
            dist = np.linalg.norm(generated_points[i+1, j] - generated_points[i, j])
            v_smoothness.append(dist)
    
    metrics['smoothness'] = {
        'horizontal_mean': np.mean(h_smoothness),
        'horizontal_std': np.std(h_smoothness),
        'vertical_mean': np.mean(v_smoothness),
        'vertical_std': np.std(v_smoothness)
    }
    
    # 5. 曲面法向量的一致性
    normals = []
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            # 计算局部法向量
            p1 = generated_points[i-1, j] - generated_points[i, j]
            p2 = generated_points[i, j-1] - generated_points[i, j]
            normal = np.cross(p1, p2)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)
    
    if normals:
        normals = np.array(normals)
        # 计算法向量的一致性（相邻法向量的角度差）
        angles = []
        for i in range(len(normals) - 1):
            cos_angle = np.clip(np.dot(normals[i], normals[i+1]), -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        metrics['normal_consistency'] = {
            'mean_angle_diff': np.mean(angles),
            'std_angle_diff': np.std(angles)
        }
    
    return metrics

def print_analysis_report(metrics: dict):
    """打印分析报告"""
    print("\n" + "="*50)
    print("曲面生成质量分析报告")
    print("="*50)
    
    print(f"\n1. 覆盖范围分析:")
    print(f"   血管范围: {metrics['vessel_bbox']['range']}")
    print(f"   曲面范围: {metrics['surface_bbox']['range']}")
    print(f"   覆盖比例: {metrics['coverage_ratio']}")
    
    print(f"\n2. 中心对齐分析:")
    print(f"   中心距离: {metrics['center_distance']:.4f}")
    
    print(f"\n3. 拟合精度分析:")
    dist_stats = metrics['min_distance_stats']
    print(f"   平均最小距离: {dist_stats['mean']:.4f}")
    print(f"   距离标准差: {dist_stats['std']:.4f}")
    print(f"   距离中位数: {dist_stats['median']:.4f}")
    print(f"   最大距离: {dist_stats['max']:.4f}")
    
    print(f"\n4. 平滑度分析:")
    smooth = metrics['smoothness']
    print(f"   水平平滑度: {smooth['horizontal_mean']:.4f} ± {smooth['horizontal_std']:.4f}")
    print(f"   垂直平滑度: {smooth['vertical_mean']:.4f} ± {smooth['vertical_std']:.4f}")
    
    if 'normal_consistency' in metrics:
        print(f"\n5. 法向量一致性:")
        normal = metrics['normal_consistency']
        print(f"   平均角度差: {normal['mean_angle_diff']:.4f} rad")
        print(f"   角度差标准差: {normal['std_angle_diff']:.4f} rad")

def comprehensive_surface_validation(tree_json: str, generated_points: np.ndarray, save_prefix: str = "surface_validation", interactive: bool = True):
    """
    综合曲面验证：可视化 + 质量分析
    
    Args:
        tree_json: 血管树json文件路径
        generated_points: 生成的曲面点，形状为(grid_size, grid_size, 3)
        save_prefix: 保存文件的前缀
        interactive: 是否显示交互式3D界面
    """
    print("开始综合曲面验证...")
    
    # 1. 可视化
    print("1. 生成可视化图像...")
    visualize_generated_surface(tree_json, generated_points, f"{save_prefix}_visualization.png", interactive=interactive)
    
    # 2. 质量分析
    print("2. 进行质量分析...")
    metrics = analyze_surface_quality(tree_json, generated_points)
    
    # 3. 打印报告
    print_analysis_report(metrics)
    
    # 4. 保存分析结果
    import json
    
    def convert_numpy_to_serializable(obj):
        """递归转换numpy对象为可序列化格式"""
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:  # 标量
                return float(obj)
            else:  # 多维数组
                return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_serializable(item) for item in obj]
        else:
            return obj
    
    with open(f"{save_prefix}_metrics.json", 'w') as f:
        serializable_metrics = convert_numpy_to_serializable(metrics)
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\n验证完成！结果已保存:")
    print(f"- 可视化图像: {save_prefix}_visualization.png")
    print(f"- 质量指标: {save_prefix}_metrics.json")
    
    if interactive:
        print("交互式3D界面已显示，您可以从多个视角查看生成的曲面效果")
    
    return metrics

def visualize_optimal_plane(tree_json: str, grid_size=32, point_spacing=0.2, save_path: str = None, interactive: bool = True):
    """
    可视化找到的最优初始平面和初始网格点分布
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    with open(tree_json,'r') as fp:
        td=json.load(fp)
    
    # 获取主干点和分支点
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
    all_branch_pts = np.vstack([br1_pts, br2_pts])
    
    # 1. 计算所有点的中心作为平面中心
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    plane_center = all_points.mean(axis=0)
    
    # 2. 找到经过分支最多的平面（分支点的主平面）
    branch_center = all_branch_pts.mean(axis=0)
    branch_centered = all_branch_pts - branch_center
    
    # 计算分支点的主平面法向量
    if len(branch_centered) >= 3:
        branch_cov = np.cov(branch_centered.T)
        branch_eigenvals, branch_eigenvecs = np.linalg.eigh(branch_cov)
        branch_idx = np.argsort(branch_eigenvals)[::-1]
        
        # 分支主平面由前两个主成分确定，法向量是第三个主成分
        branch_plane_normal = branch_eigenvecs[:, branch_idx[2]]  # 最小特征值对应方向
    else:
        # 如果分支点太少，使用默认法向量
        branch_plane_normal = np.array([0, 0, 1])
    
    # 确保法向量是单位向量
    branch_plane_normal = branch_plane_normal / np.linalg.norm(branch_plane_normal)
    
    # 3. 找到与分支平面垂直且经过主干最多的平面
    trunk_center = trunk_pts.mean(axis=0)
    trunk_centered = trunk_pts - trunk_center
    
    # 将主干点投影到垂直于分支平面法向量的空间中
    projection_matrix = np.eye(3) - np.outer(branch_plane_normal, branch_plane_normal)
    trunk_projected = trunk_centered @ projection_matrix.T
    
    # 在投影空间中找主干点的主方向
    if len(trunk_projected) >= 2:
        trunk_proj_cov = np.cov(trunk_projected.T)
        trunk_eigenvals, trunk_eigenvecs = np.linalg.eigh(trunk_proj_cov)
        trunk_idx = np.argsort(trunk_eigenvals)[::-1]
        
        # 主干在投影空间中的主方向
        trunk_main_dir_projected = trunk_eigenvecs[:, trunk_idx[0]]
    else:
        trunk_main_dir_projected = np.array([1, 0, 0])
        trunk_main_dir_projected = trunk_main_dir_projected - np.dot(trunk_main_dir_projected, branch_plane_normal) * branch_plane_normal
    
    # 确保主干主方向是单位向量
    trunk_main_dir_projected = trunk_main_dir_projected / (np.linalg.norm(trunk_main_dir_projected) + 1e-8)
    
    # 4. 构建初始平面的坐标系
    plane_normal = branch_plane_normal
    u_axis = trunk_main_dir_projected
    v_axis = np.cross(plane_normal, u_axis)
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-8)
    u_axis = np.cross(v_axis, plane_normal)
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
    
    # 5. 在平面上生成32x32的正方形网格，间隔为point_spacing
    grid_extent = (grid_size - 1) * point_spacing / 2
    u_coords = np.linspace(-grid_extent, grid_extent, grid_size)
    v_coords = np.linspace(-grid_extent, grid_extent, grid_size)
    U, V = np.meshgrid(u_coords, v_coords)
    
    # 将网格点转换到3D空间
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            point_3d = plane_center + U[i,j] * u_axis + V[i,j] * v_axis
            grid_points.append(point_3d)
    
    grid_points = np.array(grid_points)
    
    # 构建分支平面的坐标系用于可视化
    if abs(np.dot(branch_plane_normal, np.array([1, 0, 0]))) < 0.9:
        branch_u = np.cross(branch_plane_normal, np.array([1, 0, 0]))
    else:
        branch_u = np.cross(branch_plane_normal, np.array([0, 1, 0]))
    branch_u = branch_u / np.linalg.norm(branch_u)
    branch_v = np.cross(branch_plane_normal, branch_u)
    branch_v = branch_v / np.linalg.norm(branch_v)
    
    # 可视化
    fig = plt.figure(figsize=(20, 6))
    
    # 子图1: 分支主平面分析
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(*trunk_pts.T, c='blue', s=3, alpha=0.8, label='主干')
    ax1.scatter(*br1_pts.T, c='green', s=3, alpha=0.8, label='分支1')
    ax1.scatter(*br2_pts.T, c='red', s=3, alpha=0.8, label='分支2')
    
    # 绘制分支主平面
    plane_size = grid_extent * 1.5
    plane_u = np.linspace(-plane_size, plane_size, 20)
    plane_v = np.linspace(-plane_size, plane_size, 20)
    Plane_U, Plane_V = np.meshgrid(plane_u, plane_v)
    
    branch_plane_points = np.zeros((20, 20, 3))
    for i in range(20):
        for j in range(20):
            branch_plane_points[i, j] = branch_center + Plane_U[i,j] * branch_u + Plane_V[i,j] * branch_v
    
    ax1.plot_surface(branch_plane_points[:,:,0], branch_plane_points[:,:,1], branch_plane_points[:,:,2], 
                    alpha=0.3, color='cyan')
    ax1.set_title('分支主平面\n(经过分支最多的平面)')
    ax1.legend()
    ax1.set_axis_off()
    
    # 子图2: 初始平面位置
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(*trunk_pts.T, c='blue', s=3, alpha=0.8, label='主干')
    ax2.scatter(*br1_pts.T, c='green', s=3, alpha=0.8, label='分支1')
    ax2.scatter(*br2_pts.T, c='red', s=3, alpha=0.8, label='分支2')
    
    # 绘制初始平面
    initial_plane_points = np.zeros((20, 20, 3))
    for i in range(20):
        for j in range(20):
            initial_plane_points[i, j] = plane_center + Plane_U[i,j] * u_axis + Plane_V[i,j] * v_axis
    
    ax2.plot_surface(initial_plane_points[:,:,0], initial_plane_points[:,:,1], initial_plane_points[:,:,2], 
                    alpha=0.4, color='yellow')
    
    # 绘制分支平面（半透明对比）
    ax2.plot_surface(branch_plane_points[:,:,0], branch_plane_points[:,:,1], branch_plane_points[:,:,2], 
                    alpha=0.1, color='cyan')
    
    ax2.set_title('初始平面\n(与分支平面垂直，经过主干最多)')
    ax2.legend()
    ax2.set_axis_off()
    
    # 子图3: 初始网格点
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(*trunk_pts.T, c='blue', s=2, alpha=0.5, label='主干')
    ax3.scatter(*br1_pts.T, c='green', s=2, alpha=0.5, label='分支1')
    ax3.scatter(*br2_pts.T, c='red', s=2, alpha=0.5, label='分支2')
    ax3.scatter(*grid_points.T, c='orange', s=5, alpha=0.8, label='初始网格点')
    
    # 绘制网格线
    grid_reshaped = grid_points.reshape(grid_size, grid_size, 3)
    for i in range(0, grid_size, 4):
        line_points = grid_reshaped[i, :, :]
        ax3.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
               color='red', alpha=0.7, linewidth=1)
    for j in range(0, grid_size, 4):
        line_points = grid_reshaped[:, j, :]
        ax3.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
               color='red', alpha=0.7, linewidth=1)
    
    ax3.set_title(f'初始网格: {grid_size}×{grid_size}\n间距: {point_spacing}')
    ax3.legend()
    ax3.set_axis_off()
    
    # 子图4: 平面视图（从分支平面法向量方向看）
    ax4 = fig.add_subplot(144)
    
    # 将所有点投影到初始平面上
    trunk_proj = np.array([np.dot(pt - plane_center, u_axis) for pt in trunk_pts]), \
                 np.array([np.dot(pt - plane_center, v_axis) for pt in trunk_pts])
    br1_proj = np.array([np.dot(pt - plane_center, u_axis) for pt in br1_pts]), \
               np.array([np.dot(pt - plane_center, v_axis) for pt in br1_pts])
    br2_proj = np.array([np.dot(pt - plane_center, u_axis) for pt in br2_pts]), \
               np.array([np.dot(pt - plane_center, v_axis) for pt in br2_pts])
    
    ax4.scatter(*trunk_proj, c='blue', s=3, alpha=0.6, label='主干投影')
    ax4.scatter(*br1_proj, c='green', s=3, alpha=0.6, label='分支1投影')
    ax4.scatter(*br2_proj, c='red', s=3, alpha=0.6, label='分支2投影')
    ax4.scatter(U.flatten(), V.flatten(), c='orange', s=8, alpha=0.8, label='网格点')
    
    # 绘制网格线
    for i in range(0, grid_size, 4):
        ax4.plot(U[i, :], V[i, :], color='red', alpha=0.5, linewidth=1)
    for j in range(0, grid_size, 4):
        ax4.plot(U[:, j], V[:, j], color='red', alpha=0.5, linewidth=1)
    
    ax4.set_title('初始平面视图')
    ax4.set_xlabel('U轴（主干主方向）')
    ax4.set_ylabel('V轴')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # 统一设置3D视图范围
    all_points_vis = np.vstack([trunk_pts, br1_pts, br2_pts, grid_points])
    center = all_points_vis.mean(axis=0)
    range_val = 30
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
    
    plt.tight_layout()
    
    # 保存图片（如果需要）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"初始平面可视化已保存到: {save_path}")
    
    # 显示交互式界面（如果需要）
    if interactive:
        print("正在打开初始平面分析的交互式3D界面...")
        print("您可以从多个视角查看初始平面的构建过程")
        plt.show()
    else:
        plt.close()
    
    print(f"初始平面信息:")
    print(f"  平面中心: {plane_center}")
    print(f"  分支平面法向量: {branch_plane_normal}")
    print(f"  主干主方向（投影后）: {trunk_main_dir_projected}")
    print(f"  初始平面法向量: {plane_normal}")
    print(f"  网格尺寸: {grid_extent*2:.2f} × {grid_extent*2:.2f}")
    print(f"  点间距: {point_spacing}")
    
    # 验证垂直性
    dot_product = np.dot(plane_normal, branch_plane_normal)
    print(f"  平面垂直性验证: {abs(dot_product):.6f} (接近0表示垂直)")
    
    return plane_center, plane_normal, u_axis, v_axis, grid_points

def visualize_training_target_surface(tree_json: str, grid_size=32, point_spacing=0.2, save_path: str = None, interactive: bool = True):
    """
    可视化训练目标曲面（以中轴线为骨架的曲面）
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    with open(tree_json,'r') as fp:
        td=json.load(fp)
    
    # 获取主干和分支点
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
    
    # 计算两个分支对应点的中点，构建中轴线
    min_len = min(len(br1_pts), len(br2_pts))
    if min_len > 0:
        br1_sampled = br1_pts[:min_len]
        br2_sampled = br2_pts[:min_len]
        midpoints = (br1_sampled + br2_sampled) / 2.0
    else:
        midpoints = np.array([trunk_pts[-1]])
    
    # 构建中轴线：主干点 + 分支中点
    centerline_points = np.vstack([trunk_pts, midpoints])
    
    # 对中轴线点进行排序，确保连续性
    centerline_center = centerline_points.mean(axis=0)
    centerline_centered = centerline_points - centerline_center
    
    # 使用PCA找到中轴线的主方向
    cov_matrix = np.cov(centerline_centered.T)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    idx = np.argsort(eigenvals)[::-1]
    main_direction = eigenvecs[:, idx[0]]  # 中轴线主方向
    
    # 将中轴线点投影到主方向上并排序
    projections = np.dot(centerline_centered, main_direction)
    sorted_indices = np.argsort(projections)
    sorted_centerline = centerline_points[sorted_indices]
    
    
    
    # 生成训练目标曲面
    temp_dataset = TempDataset()
    target_surface = temp_dataset._generate_surface_grid(
        sorted_centerline, main_direction, grid_size, point_spacing
    )
    
    # 可视化
    fig = plt.figure(figsize=(20, 6))
    
    # 子图1: 中轴线和血管点
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(*trunk_pts.T, c='blue', s=3, alpha=0.8, label='主干')
    ax1.scatter(*br1_pts.T, c='green', s=3, alpha=0.8, label='分支1')
    ax1.scatter(*br2_pts.T, c='red', s=3, alpha=0.8, label='分支2')
    ax1.scatter(*midpoints.T, c='purple', s=5, alpha=0.9, label='分支中点')
    ax1.plot(*sorted_centerline.T, 'orange', linewidth=3, alpha=0.8, label='中轴线')
    ax1.set_title('中轴线构建')
    ax1.legend()
    ax1.set_axis_off()
    
    # 子图2: 目标曲面
    ax2 = fig.add_subplot(142, projection='3d')
    X, Y, Z = target_surface[:, :, 0], target_surface[:, :, 1], target_surface[:, :, 2]
    ax2.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
    ax2.plot(*sorted_centerline.T, 'red', linewidth=3, alpha=1.0, label='中轴线')
    ax2.set_title('训练目标曲面')
    ax2.legend()
    ax2.set_axis_off()
    
    # 子图3: 曲面+血管叠加
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(*trunk_pts.T, c='blue', s=2, alpha=0.6, label='主干')
    ax3.scatter(*br1_pts.T, c='green', s=2, alpha=0.6, label='分支1')
    ax3.scatter(*br2_pts.T, c='red', s=2, alpha=0.6, label='分支2')
    ax3.plot_surface(X, Y, Z, alpha=0.4, color='orange')
    ax3.plot(*sorted_centerline.T, 'purple', linewidth=2, alpha=0.8)
    
    # 绘制网格线显示结构
    for i in range(0, grid_size, 4):
        ax3.plot(X[i, :], Y[i, :], Z[i, :], 'k-', alpha=0.3, linewidth=0.5)
    for j in range(0, grid_size, 4):
        ax3.plot(X[:, j], Y[:, j], Z[:, j], 'k-', alpha=0.3, linewidth=0.5)
    
    ax3.set_title('曲面与血管叠加')
    ax3.legend()
    ax3.set_axis_off()
    
    # 子图4: 网格点分布
    ax4 = fig.add_subplot(144, projection='3d')
    surface_points_flat = target_surface.reshape(-1, 3)
    ax4.scatter(*surface_points_flat.T, c='orange', s=2, alpha=0.8, label='曲面网格点')
    ax4.plot(*sorted_centerline.T, 'red', linewidth=2, alpha=0.8, label='中轴线')
    
    # 显示网格结构
    for i in range(0, grid_size, 2):
        line_points = target_surface[i, :, :]
        ax4.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
               'b-', alpha=0.5, linewidth=0.5)
    for j in range(0, grid_size, 2):
        line_points = target_surface[:, j, :]
        ax4.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
               'g-', alpha=0.5, linewidth=0.5)
    
    ax4.set_title(f'网格点分布\n{grid_size}×{grid_size}, 间距{point_spacing}')
    ax4.legend()
    ax4.set_axis_off()
    
    # 统一设置视图范围
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts, surface_points_flat])
    center = all_points.mean(axis=0)
    range_val = 30
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
    
    plt.tight_layout()
    
    # 保存图片（如果需要）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练目标曲面可视化已保存到: {save_path}")
    
    # 显示交互式界面（如果需要）
    if interactive:
        print("正在打开训练目标曲面的交互式3D界面...")
        print("您可以从多个视角查看训练目标曲面的构建过程")
        plt.show()
    else:
        plt.close()
    
    print(f"训练目标曲面信息:")
    print(f"  中轴线点数: {len(sorted_centerline)}")
    print(f"  曲面网格: {grid_size}×{grid_size}")
    print(f"  点间距: {point_spacing}")
    print(f"  曲面范围: {(grid_size-1)*point_spacing:.2f}")
    print(f"  总点数: {grid_size*grid_size}")
    
    return target_surface, sorted_centerline

# --------- 快速演示交互式可视化 ---------

def quick_demo_interactive_visualization(tree_json: str):
    """
    快速演示交互式3D可视化功能
    """
    print("=== 交互式3D可视化演示 ===")
    print("即将依次展示4个交互式3D界面，您可以:")
    print("- 用鼠标拖拽旋转视角")
    print("- 用滚轮缩放")
    print("- 关闭当前窗口查看下一个界面")
    print("- 按Ctrl+C中断演示")
    
    input("按回车键开始演示...")
    
    try:
        # 1. 展示最优初始平面
        print("\n1. 最优初始平面分析")
        # visualize_optimal_plane(tree_json, grid_size=16, point_spacing=0.3, 
        #                        save_path=None, interactive=True)
        
        # 2. 展示训练目标曲面  
        print("\n2. 训练目标曲面")
        target_surface, _ = visualize_training_target_surface(
            tree_json, grid_size=16, point_spacing=0.3,
            save_path=None, interactive=True
        )
        
        # 3. 如果有训练好的模型，展示生成结果
        print("\n3. 基于中轴线的分隔曲面")
        print("注意：此演示使用基于中轴线的模拟数据，实际使用时需要训练好的模型")
        
        # 读取血管树数据并计算中轴线
        with open(tree_json, 'r') as fp:
            td = json.load(fp)
        
        trunk_pts, br1_pts, br2_pts = safe_find_max_points_branches(td)
        
        # 计算分支中点构建中轴线
        min_len = min(len(br1_pts), len(br2_pts))
        if min_len > 0:
            br1_sampled = br1_pts[:min_len]
            br2_sampled = br2_pts[:min_len]
            midpoints = (br1_sampled + br2_sampled) / 2.0
        else:
            midpoints = np.array([trunk_pts[-1]])
        
        # 构建中轴线：主干点 + 分支中点
        centerline_points = np.vstack([trunk_pts, midpoints])
        centerline_center = centerline_points.mean(axis=0)
        centerline_centered = centerline_points - centerline_center
        
        # 使用PCA找到中轴线的主方向
        cov_matrix = np.cov(centerline_centered.T)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        idx = np.argsort(eigenvals)[::-1]
        main_direction = eigenvecs[:, idx[0]]  # 中轴线主方向
        
        # 将中轴线点投影到主方向上并排序
        projections = np.dot(centerline_centered, main_direction)
        sorted_indices = np.argsort(projections)
        sorted_centerline = centerline_points[sorted_indices]
        
        print(f"中轴线信息: {len(sorted_centerline)} 个点，主方向: {main_direction}")
        
        # 创建基于中轴线的分隔曲面，有效分离两个分支
        grid_size = 16
        demo_surface = np.zeros((grid_size, grid_size, 3))
        
        # 分析两个分支的分布
        br1_center = br1_pts.mean(axis=0)
        br2_center = br2_pts.mean(axis=0)
        centerline_center = sorted_centerline.mean(axis=0)
        
        # 方法：使用两个分支中心连线的垂直平分面作为分隔基准
        branch_connection = br2_center - br1_center  # 从分支1指向分支2的向量
        branch_midpoint = (br1_center + br2_center) / 2  # 两分支的中点
        
        # 分隔面法向量：两个分支中心的连线方向
        if np.linalg.norm(branch_connection) > 1e-6:
            separation_normal = branch_connection / np.linalg.norm(branch_connection)
        else:
            # 如果两个分支中心重合，使用分支点的主成分分析
            all_branch_pts = np.vstack([br1_pts, br2_pts])
            branch_centered = all_branch_pts - all_branch_pts.mean(axis=0)
            if len(branch_centered) >= 3:
                branch_cov = np.cov(branch_centered.T)
                eigenvals, eigenvecs = np.linalg.eigh(branch_cov)
                idx = np.argsort(eigenvals)[::-1]
                separation_normal = eigenvecs[:, idx[0]]  # 最大变化方向
            else:
                separation_normal = np.array([1, 0, 0])
        
        # 构建分隔曲面的坐标系
        # u轴：中轴线主方向在分隔面上的投影（沿血管走向）
        centerline_proj = main_direction - np.dot(main_direction, separation_normal) * separation_normal
        if np.linalg.norm(centerline_proj) < 1e-6:
            # 如果中轴线与分隔面法向量平行，选择其他方向
            temp_vec = np.array([0, 0, 1]) if abs(separation_normal[2]) < 0.9 else np.array([1, 0, 0])
            centerline_proj = temp_vec - np.dot(temp_vec, separation_normal) * separation_normal
        
        u_axis = centerline_proj / np.linalg.norm(centerline_proj)
        
        # v轴：垂直于u轴和法向量
        v_axis = np.cross(separation_normal, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)
        
        # 验证分离效果
        br1_to_midpoint = br1_center - branch_midpoint
        br2_to_midpoint = br2_center - branch_midpoint
        br1_side = np.dot(br1_to_midpoint, separation_normal)
        br2_side = np.dot(br2_to_midpoint, separation_normal)
        
        print(f"分支分离分析:")
        print(f"  分支1中心: {br1_center}")
        print(f"  分支2中心: {br2_center}")
        print(f"  分支中点: {branch_midpoint}")
        print(f"  分支连线向量: {branch_connection}")
        print(f"  分隔面法向量: {separation_normal}")
        print(f"  分支1到分隔面距离: {br1_side:.3f}")
        print(f"  分支2到分隔面距离: {br2_side:.3f}")
        print(f"  分支是否在两侧: {br1_side * br2_side < 0}")
        
        # 确保分隔面经过中轴线附近
        # 将分隔面中心设置为中轴线中心和分支中点的加权平均
        separation_center = 0.7 * centerline_center + 0.3 * branch_midpoint
        
        # 计算曲面尺寸
        centerline_length = np.linalg.norm(sorted_centerline[-1] - sorted_centerline[0])
        branch_separation = np.linalg.norm(branch_connection)
        all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
        points_range = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        
        # u方向：沿血管走向，覆盖整个中轴线
        surface_extent_u = max(centerline_length * 1.5, points_range.max() * 1.0)
        # v方向：垂直方向，足够覆盖分支范围
        surface_extent_v = max(branch_separation * 2.0, points_range.max() * 0.8)
        
        print(f"  分隔面中心: {separation_center}")
        print(f"  曲面尺寸: u={surface_extent_u:.2f}, v={surface_extent_v:.2f}")
        
        # 生成分隔曲面点
        for i in range(grid_size):
            for j in range(grid_size):
                # 参数化
                u_param = (i / (grid_size - 1) - 0.5) * 2  # -1 到 1
                v_param = (j / (grid_size - 1) - 0.5) * 2  # -1 到 1
                
                # 在分隔面上生成点
                offset_u = u_param * surface_extent_u * 0.4
                offset_v = v_param * surface_extent_v * 0.4
                
                # 基准位置：分隔面中心
                base_point = separation_center
                
                # 如果要让曲面更贴合中轴线，可以在u方向上插值到中轴线
                if len(sorted_centerline) > 1:
                    # 将u参数映射到中轴线
                    centerline_param = (u_param + 1) / 2  # 转换到0-1
                    centerline_param = np.clip(centerline_param, 0, 1)
                    
                    if centerline_param <= 0:
                        centerline_point = sorted_centerline[0]
                    elif centerline_param >= 1:
                        centerline_point = sorted_centerline[-1]
                    else:
                        idx_float = centerline_param * (len(sorted_centerline) - 1)
                        idx_low = int(idx_float)
                        idx_high = min(idx_low + 1, len(sorted_centerline) - 1)
                        alpha = idx_float - idx_low
                        centerline_point = (1 - alpha) * sorted_centerline[idx_low] + alpha * sorted_centerline[idx_high]
                    
                    # 将基准点调整为中轴线点在分隔面上的投影
                    centerline_to_center = centerline_point - separation_center
                    # 将中轴线点投影到分隔面上
                    projected_offset = centerline_to_center - np.dot(centerline_to_center, separation_normal) * separation_normal
                    base_point = separation_center + projected_offset * 0.8  # 部分跟随中轴线
                
                # 添加轻微的曲面弯曲，使其更自然
                curvature = 0.1 * np.sin(u_param * np.pi) * np.cos(v_param * np.pi * 2)
                
                # 计算最终位置
                point_3d = (base_point + 
                           offset_u * u_axis + 
                           offset_v * v_axis + 
                           curvature * separation_normal)
                
                demo_surface[i, j] = point_3d
        
        print(f"分隔曲面信息:")
        print(f"  曲面中心: {demo_surface.mean(axis=(0,1))}")
        print(f"  u轴(沿血管): {u_axis}")
        print(f"  v轴(垂直): {v_axis}")
        print(f"  法向量(分离方向): {separation_normal}")
        print(f"  目标：将绿色和红色分支分离到曲面两侧")
        
        visualize_generated_surface(tree_json, demo_surface, 
                                  save_path=None, interactive=True)
        
        # 4. 详细验证分离效果
        print("\n4. 分离效果详细验证")
        print("显示详细的分离效果分析，包括统计数据和可视化")
        visualize_separation_effect(tree_json, demo_surface, interactive=True)
        
        print("\n演示完成！")
        print("您已经查看了4个交互式3D界面:")
        print("1. 最优初始平面分析 - 显示如何选择初始平面")
        print("2. 训练目标曲面 - 显示以中轴线为骨架的目标曲面") 
        print("3. 基于中轴线的分隔曲面 - 显示用于分离分支的曲面")
        print("4. 分离效果详细验证 - 显示分离效果的定量分析")
        print("所有界面都支持交互式3D查看，便于从多角度理解算法原理。")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")

if __name__=='__main__':
    import glob
    files = glob.glob('tree_*.json')
    if len(files):
        print(f"发现 {len(files)} 个血管树文件")
        
        # 添加快速演示选项
        print("\n选择运行模式:")
        print("1. 快速交互式3D可视化演示")
        print("   - 无需训练，立即查看4个交互式3D界面")
        print("   - 包含：初始平面分析、目标曲面、分隔曲面、分离效果验证")
        print("   - 推荐用于理解算法原理和调试")
        print("2. 完整训练和生成流程")
        print("   - 包含完整的训练过程（训练轮数已调整为500轮以便快速测试）")
        print("   - 最终会显示实际训练的结果")
        
        choice = input("请输入选择 (1 或 2，默认为 2): ").strip()
        
        if choice == "1":
            print("启动快速演示模式...")
            quick_demo_interactive_visualization(files[0])
            exit()
        else:
            print("启动完整训练流程...")
        
        grid_size = 32
        point_spacing = 0.2  # 可调节的点间距
        
        print("=== 可视化最优初始平面 ===")
        # visualize_optimal_plane(files[0], grid_size=grid_size, point_spacing=point_spacing, 
        #                        save_path="optimal_plane_visualization.png", interactive=True)
        
        print("\n=== 可视化训练目标曲面 ===")
        target_surface, centerline = visualize_training_target_surface(
            files[0], grid_size=grid_size, point_spacing=point_spacing, 
            save_path="training_target_surface.png", interactive=True
        )
        
        print("\n=== 开始训练扩散模型 ===")
        model, betas = train_tree_diffusion(files, epochs=100000, device='cpu', grid_size=grid_size, point_spacing=point_spacing)
        
        print("\n=== 生成曲面 ===")
        pred = denoise_with_tree(files[0], model, betas, device='cpu', grid_size=grid_size, point_spacing=point_spacing)
        pred_gif = denoise_with_gif(files[0], model, betas, gif_path='denoise.gif', device='cpu', 
                                  grid_size=grid_size, point_spacing=point_spacing)
        print('Predicted plane points shape:', pred.shape)  # (32, 32, 3)
        print('Final points center:', pred.mean(axis=(0,1))) 
        
        # 新增：综合验证生成的曲面
        print("\n=== 验证生成的曲面 ===")
        comprehensive_surface_validation(files[0], pred, "trained_surface", interactive=True)

def visualize_separation_effect(tree_json: str, separation_surface: np.ndarray, interactive: bool = True):
    """
    专门用于验证和可视化分隔曲面的分离效果
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 读取原始血管数据
    with open(tree_json, 'r') as fp:
        td = json.load(fp)
    
    trunk_pts, br1_pts, br2_pts = safe_find_max_points_branches(td)
    
    # 计算分隔面参数
    br1_center = br1_pts.mean(axis=0)
    br2_center = br2_pts.mean(axis=0)
    branch_midpoint = (br1_center + br2_center) / 2
    branch_connection = br2_center - br1_center
    
    if np.linalg.norm(branch_connection) > 1e-6:
        separation_normal = branch_connection / np.linalg.norm(branch_connection)
    else:
        separation_normal = np.array([1, 0, 0])
    
    # 计算曲面中心
    surface_center = separation_surface.mean(axis=(0,1))
    
    # 创建大尺寸图形以便详细观察
    fig = plt.figure(figsize=(20, 8))
    
    # 子图1: 总体视图
    ax1 = fig.add_subplot(141, projection='3d')
    
    # 绘制血管点
    ax1.scatter(*trunk_pts.T, c='blue', s=3, alpha=0.8, label='主干', marker='o')
    ax1.scatter(*br1_pts.T, c='green', s=5, alpha=0.9, label='分支1', marker='^')
    ax1.scatter(*br2_pts.T, c='red', s=5, alpha=0.9, label='分支2', marker='s')
    
    # 绘制分支中心
    ax1.scatter(*br1_center, c='darkgreen', s=100, marker='*', label='分支1中心')
    ax1.scatter(*br2_center, c='darkred', s=100, marker='*', label='分支2中心')
    ax1.scatter(*branch_midpoint, c='purple', s=100, marker='D', label='分支中点')
    
    # 绘制分隔曲面
    X, Y, Z = separation_surface[:, :, 0], separation_surface[:, :, 1], separation_surface[:, :, 2]
    ax1.plot_surface(X, Y, Z, alpha=0.4, color='orange', label='分隔曲面')
    
    # 绘制分支连线
    ax1.plot([br1_center[0], br2_center[0]], 
             [br1_center[1], br2_center[1]], 
             [br1_center[2], br2_center[2]], 
             'purple', linewidth=3, alpha=0.8, label='分支连线')
    
    ax1.set_title('分支分离总视图')
    ax1.legend()
    ax1.set_axis_off()
    
    # 子图2: 沿分隔面法向量的侧视图
    ax2 = fig.add_subplot(142, projection='3d')
    
    # 计算点到分隔面的距离
    br1_distances = [np.dot(pt - surface_center, separation_normal) for pt in br1_pts]
    br2_distances = [np.dot(pt - surface_center, separation_normal) for pt in br2_pts]
    trunk_distances = [np.dot(pt - surface_center, separation_normal) for pt in trunk_pts]
    
    # 根据到分隔面的距离给点着色
    br1_colors = ['lightgreen' if d < 0 else 'darkgreen' for d in br1_distances]
    br2_colors = ['lightcoral' if d < 0 else 'darkred' for d in br2_distances]
    
    for i, (pt, color) in enumerate(zip(br1_pts, br1_colors)):
        ax2.scatter(*pt, c=color, s=8, alpha=0.8)
    for i, (pt, color) in enumerate(zip(br2_pts, br2_colors)):
        ax2.scatter(*pt, c=color, s=8, alpha=0.8)
    
    ax2.scatter(*trunk_pts.T, c='blue', s=2, alpha=0.6)
    ax2.plot_surface(X, Y, Z, alpha=0.3, color='orange')
    
    ax2.set_title('按分隔面分侧着色\n(浅色=负侧，深色=正侧)')
    ax2.set_axis_off()
    
    # 子图3: 距离分布统计
    ax3 = fig.add_subplot(143)
    
    ax3.hist(br1_distances, bins=20, alpha=0.7, color='green', label=f'分支1 (均值:{np.mean(br1_distances):.2f})')
    ax3.hist(br2_distances, bins=20, alpha=0.7, color='red', label=f'分支2 (均值:{np.mean(br2_distances):.2f})')
    ax3.axvline(0, color='orange', linestyle='--', linewidth=2, label='分隔面')
    ax3.set_xlabel('到分隔面的距离')
    ax3.set_ylabel('点数')
    ax3.set_title('分支点到分隔面距离分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 分离效果量化
    ax4 = fig.add_subplot(144)
    
    # 计算分离效果指标
    br1_positive = sum(1 for d in br1_distances if d > 0)
    br1_negative = sum(1 for d in br1_distances if d < 0)
    br2_positive = sum(1 for d in br2_distances if d > 0)
    br2_negative = sum(1 for d in br2_distances if d < 0)
    
    # 理想情况：两个分支应该在分隔面的不同侧
    br1_majority_side = "正侧" if br1_positive > br1_negative else "负侧"
    br2_majority_side = "正侧" if br2_positive > br2_negative else "负侧"
    
    separation_quality = "良好" if br1_majority_side != br2_majority_side else "需要改进"
    
    # 绘制分离统计
    categories = ['分支1\n正侧', '分支1\n负侧', '分支2\n正侧', '分支2\n负侧']
    values = [br1_positive, br1_negative, br2_positive, br2_negative]
    colors = ['darkgreen', 'lightgreen', 'darkred', 'lightcoral']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.8)
    ax4.set_ylabel('点数')
    ax4.set_title(f'分离效果统计\n分离质量: {separation_quality}')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(value), ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    # 在图上添加分析文本
    analysis_text = f"""分离分析:
分支1: {br1_majority_side} ({br1_positive}/{br1_positive+br1_negative})
分支2: {br2_majority_side} ({br2_positive}/{br2_positive+br2_negative})
分离质量: {separation_quality}
"""
    
    fig.text(0.02, 0.02, analysis_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # 统一设置3D视图范围
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    center = all_points.mean(axis=0)
    range_val = 30
    
    for ax in [ax1, ax2]:
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
    
    plt.tight_layout()
    
    # 打印详细分析
    print("\n" + "="*60)
    print("分隔效果详细分析")
    print("="*60)
    print(f"分支1点数: {len(br1_pts)}")
    print(f"  - 在分隔面正侧: {br1_positive} 个点")
    print(f"  - 在分隔面负侧: {br1_negative} 个点")
    print(f"  - 主要分布: {br1_majority_side}")
    print(f"  - 距离均值: {np.mean(br1_distances):.3f}")
    
    print(f"\n分支2点数: {len(br2_pts)}")
    print(f"  - 在分隔面正侧: {br2_positive} 个点")
    print(f"  - 在分隔面负侧: {br2_negative} 个点")
    print(f"  - 主要分布: {br2_majority_side}")
    print(f"  - 距离均值: {np.mean(br2_distances):.3f}")
    
    print(f"\n分离效果评估:")
    print(f"  - 两分支主要分布在不同侧: {br1_majority_side != br2_majority_side}")
    print(f"  - 分离质量: {separation_quality}")
    
    if br1_majority_side == br2_majority_side:
        print(f"  - 建议: 需要调整分隔面的位置或方向")
    else:
        print(f"  - 结果: 分隔曲面成功将两个分支分离")
    
    # 显示交互式界面（如果需要）
    if interactive:
        print("\n正在打开分离效果验证的交互式界面...")
        plt.show()
    else:
        plt.close()

print("测试merge")
print("ceshi2")

# --------- SVM曲面切分 ---------
def create_svm_separation_surface(points_group1, points_group2, grid_size=32, plane_size=10.0, kernel='rbf', C=1.0):
    """
    使用SVM创建一个曲面来区分两组3D点
    
    Args:
        points_group1: 第一组点 (N1, 3)
        points_group2: 第二组点 (N2, 3)
        grid_size: 生成的曲面网格大小
        plane_size: 曲面的大小范围
        kernel: SVM核函数 ('linear', 'rbf', 'poly')
        C: SVM正则化参数
    
    Returns:
        separation_surface: 分离曲面 (grid_size, grid_size, 3)
        svm_model: 训练好的SVM模型
        separation_info: 分离信息字典
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # 准备训练数据
    X = np.vstack([points_group1, points_group2])
    y = np.hstack([np.ones(len(points_group1)), -np.ones(len(points_group2))])
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练SVM模型
    svm_model = SVC(kernel=kernel, C=C, probability=True)
    svm_model.fit(X_scaled, y)
    
    # 计算分离超平面的法向量和偏移
    if kernel == 'linear':
        # 线性核可以直接获取法向量
        normal_vector = svm_model.coef_[0]
        bias = svm_model.intercept_[0]
    else:
        # 非线性核需要近似法向量
        normal_vector, bias = _approximate_svm_normal(svm_model, X_scaled, y)
    
    # 计算两组点的中心
    center_group1 = np.mean(points_group1, axis=0)
    center_group2 = np.mean(points_group2, axis=0)
    center = (center_group1 + center_group2) / 2
    
    # 生成分离曲面
    separation_surface = _generate_separation_surface(
        center, normal_vector, bias, scaler, svm_model, 
        grid_size, plane_size, kernel
    )
    
    # 计算分离效果
    separation_info = _evaluate_separation_quality(
        points_group1, points_group2, separation_surface, svm_model, scaler
    )
    
    return separation_surface, svm_model, separation_info

def _approximate_svm_normal(svm_model, X_scaled, y):
    """
    近似计算非线性SVM的法向量
    """
    # 使用支持向量来近似法向量
    support_vectors = svm_model.support_vectors_
    support_vector_coefs = svm_model.dual_coef_[0]
    
    # 计算加权支持向量的平均方向
    weighted_sv = np.sum(support_vectors * support_vector_coefs[:, np.newaxis], axis=0)
    normal_vector = weighted_sv / (np.linalg.norm(weighted_sv) + 1e-8)
    
    # 计算偏移
    bias = svm_model.intercept_[0]
    
    return normal_vector, bias

def _generate_separation_surface(center, normal_vector, bias, scaler, svm_model, grid_size, plane_size, kernel):
    """
    生成分离曲面
    """
    # 构建垂直于法向量的两个基向量
    if abs(np.dot(normal_vector, np.array([1, 0, 0]))) < 0.9:
        base_vector = np.array([1, 0, 0])
    else:
        base_vector = np.array([0, 1, 0])
    
    u_axis = np.cross(normal_vector, base_vector)
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
    
    v_axis = np.cross(normal_vector, u_axis)
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-8)
    
    # 生成网格
    g = np.linspace(-plane_size/2, plane_size/2, grid_size)
    u, v = np.meshgrid(g, g)
    
    # 初始化曲面
    surface = np.zeros((grid_size, grid_size, 3))
    
    for i in range(grid_size):
        for j in range(grid_size):
            # 基础平面点
            base_point = center + u[i, j] * u_axis + v[i, j] * v_axis
            
            if kernel == 'linear':
                # 线性核：直接使用超平面
                # 计算到超平面的距离
                distance = np.dot(base_point, normal_vector) + bias
                # 将点投影到超平面上
                surface[i, j] = base_point - distance * normal_vector
            else:
                # 非线性核：使用SVM决策函数
                surface[i, j] = _find_decision_boundary_point(
                    base_point, normal_vector, svm_model, scaler, grid_size
                )
    
    return surface

def _find_decision_boundary_point(base_point, normal_vector, svm_model, scaler, max_iter=10):
    """
    找到决策边界上的点
    """
    current_point = base_point.copy()
    
    for _ in range(max_iter):
        # 标准化当前点
        current_point_scaled = scaler.transform(current_point.reshape(1, -1))
        
        # 计算决策函数值
        decision_value = svm_model.decision_function(current_point_scaled)[0]
        
        # 如果接近决策边界，停止迭代
        if abs(decision_value) < 1e-3:
            break
        
        # 沿法向量方向调整点
        adjustment = decision_value * normal_vector * 0.1
        current_point -= adjustment
    
    return current_point

def _evaluate_separation_quality(points_group1, points_group2, separation_surface, svm_model, scaler):
    """
    评估分离质量
    """
    # 计算分离准确率
    X = np.vstack([points_group1, points_group2])
    y_true = np.hstack([np.ones(len(points_group1)), -np.ones(len(points_group2))])
    
    X_scaled = scaler.transform(X)
    y_pred = svm_model.predict(X_scaled)
    
    accuracy = np.mean(y_pred == y_true)
    
    # 计算两组点到曲面的平均距离
    surface_center = np.mean(separation_surface, axis=(0, 1))
    
    distances_group1 = []
    distances_group2 = []
    
    for point in points_group1:
        # 找到曲面上最近的点
        distances = np.linalg.norm(separation_surface.reshape(-1, 3) - point, axis=1)
        min_distance = np.min(distances)
        distances_group1.append(min_distance)
    
    for point in points_group2:
        distances = np.linalg.norm(separation_surface.reshape(-1, 3) - point, axis=1)
        min_distance = np.min(distances)
        distances_group2.append(min_distance)
    
    avg_distance_group1 = np.mean(distances_group1)
    avg_distance_group2 = np.mean(distances_group2)
    
    # 计算分离度（两组点距离的差异）
    separation_degree = abs(avg_distance_group1 - avg_distance_group2)
    
    return {
        'accuracy': accuracy,
        'avg_distance_group1': avg_distance_group1,
        'avg_distance_group2': avg_distance_group2,
        'separation_degree': separation_degree,
        'support_vectors_count': len(svm_model.support_vectors_)
    }

def visualize_svm_separation(points_group1, points_group2, separation_surface, separation_info, 
                           save_path=None, interactive=True):
    """
    可视化SVM分离结果
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('SVM曲面分离可视化', fontsize=16)
    
    # 主视图：点云和分离曲面
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(points_group1[:, 0], points_group1[:, 1], points_group1[:, 2], 
               c='red', s=20, alpha=0.7, label='组1')
    ax1.scatter(points_group2[:, 0], points_group2[:, 1], points_group2[:, 2], 
               c='blue', s=20, alpha=0.7, label='组2')
    ax1.plot_surface(separation_surface[:, :, 0], separation_surface[:, :, 1], separation_surface[:, :, 2], 
                    alpha=0.3, color='green', label='分离曲面')
    ax1.set_title('SVM分离结果')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()
    
    # 分离曲面细节
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(separation_surface[:, :, 0], separation_surface[:, :, 1], separation_surface[:, :, 2], 
                    alpha=0.8, cmap='viridis')
    ax2.set_title('分离曲面')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # 距离分布
    ax3 = fig.add_subplot(2, 3, 3)
    distances_group1 = []
    distances_group2 = []
    
    for point in points_group1:
        distances = np.linalg.norm(separation_surface.reshape(-1, 3) - point, axis=1)
        distances_group1.append(np.min(distances))
    
    for point in points_group2:
        distances = np.linalg.norm(separation_surface.reshape(-1, 3) - point, axis=1)
        distances_group2.append(np.min(distances))
    
    ax3.hist(distances_group1, bins=20, alpha=0.7, label='组1距离', color='red')
    ax3.hist(distances_group2, bins=20, alpha=0.7, label='组2距离', color='blue')
    ax3.set_title('到分离曲面的距离分布')
    ax3.set_xlabel('距离'); ax3.set_ylabel('频次')
    ax3.legend()
    
    # 统计信息
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    stats_text = f"""
    分离质量统计:
    
    准确率: {separation_info['accuracy']:.3f}
    组1平均距离: {separation_info['avg_distance_group1']:.3f}
    组2平均距离: {separation_info['avg_distance_group2']:.3f}
    分离度: {separation_info['separation_degree']:.3f}
    支持向量数量: {separation_info['support_vectors_count']}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # 投影视图
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(points_group1[:, 0], points_group1[:, 1], c='red', s=20, alpha=0.7, label='组1')
    ax5.scatter(points_group2[:, 0], points_group2[:, 1], c='blue', s=20, alpha=0.7, label='组2')
    ax5.set_title('XY平面投影')
    ax5.set_xlabel('X'); ax5.set_ylabel('Y')
    ax5.legend()
    
    # 3D投影
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.scatter(points_group1[:, 0], points_group1[:, 1], points_group1[:, 2], 
               c='red', s=20, alpha=0.7, label='组1')
    ax6.scatter(points_group2[:, 0], points_group2[:, 1], points_group2[:, 2], 
               c='blue', s=20, alpha=0.7, label='组2')
    ax6.set_title('3D点云分布')
    ax6.set_xlabel('X'); ax6.set_ylabel('Y'); ax6.set_zlabel('Z')
    ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SVM分离可视化已保存: {save_path}")
    
    if interactive:
        plt.show()
    else:
        plt.close()

# --------- 其他分离方法 ---------
def create_lda_separation_surface(points_group1, points_group2, grid_size=32, plane_size=10.0):
    """
    使用线性判别分析(LDA)创建分离曲面
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import numpy as np
    
    # 准备数据
    X = np.vstack([points_group1, points_group2])
    y = np.hstack([np.ones(len(points_group1)), np.zeros(len(points_group2))])
    
    # 训练LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # 获取分离超平面
    normal_vector = lda.coef_[0]
    bias = lda.intercept_[0]
    
    # 生成分离曲面
    center = (np.mean(points_group1, axis=0) + np.mean(points_group2, axis=0)) / 2
    separation_surface = _generate_lda_surface(center, normal_vector, bias, grid_size, plane_size)
    
    return separation_surface, lda

def create_kmeans_separation_surface(points_group1, points_group2, grid_size=32, plane_size=10.0):
    """
    使用K-means聚类创建分离曲面
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    # 合并所有点
    all_points = np.vstack([points_group1, points_group2])
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(all_points)
    
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    
    # 计算分离超平面
    normal_vector = centers[1] - centers[0]
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    bias = -np.dot(normal_vector, (centers[0] + centers[1]) / 2)
    
    # 生成分离曲面
    center = (centers[0] + centers[1]) / 2
    separation_surface = _generate_lda_surface(center, normal_vector, bias, grid_size, plane_size)
    
    return separation_surface, kmeans

def _generate_lda_surface(center, normal_vector, bias, grid_size, plane_size):
    """
    生成LDA分离曲面
    """
    # 构建垂直于法向量的两个基向量
    if abs(np.dot(normal_vector, np.array([1, 0, 0]))) < 0.9:
        base_vector = np.array([1, 0, 0])
    else:
        base_vector = np.array([0, 1, 0])
    
    u_axis = np.cross(normal_vector, base_vector)
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
    
    v_axis = np.cross(normal_vector, u_axis)
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-8)
    
    # 生成网格
    g = np.linspace(-plane_size/2, plane_size/2, grid_size)
    u, v = np.meshgrid(g, g)
    
    # 生成曲面
    surface = np.zeros((grid_size, grid_size, 3))
    
    for i in range(grid_size):
        for j in range(grid_size):
            base_point = center + u[i, j] * u_axis + v[i, j] * v_axis
            distance = np.dot(base_point, normal_vector) + bias
            surface[i, j] = base_point - distance * normal_vector
    
    return surface

def demo_separation_methods():
    """
    演示不同分离方法的效果
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print("=== 曲面分离方法演示 ===")
    
    # 生成两组分离的点云数据
    np.random.seed(42)
    
    # 组1：球形分布
    n1 = 200
    r1 = 2.0
    theta1 = np.random.uniform(0, 2*np.pi, n1)
    phi1 = np.random.uniform(0, np.pi, n1)
    x1 = r1 * np.sin(phi1) * np.cos(theta1) + np.random.normal(0, 0.3, n1)
    y1 = r1 * np.sin(phi1) * np.sin(theta1) + np.random.normal(0, 0.3, n1)
    z1 = r1 * np.cos(phi1) + np.random.normal(0, 0.3, n1)
    points_group1 = np.column_stack([x1, y1, z1])
    
    # 组2：椭球形分布
    n2 = 200
    r2 = 3.0
    theta2 = np.random.uniform(0, 2*np.pi, n2)
    phi2 = np.random.uniform(0, np.pi, n2)
    x2 = r2 * np.sin(phi2) * np.cos(theta2) + np.random.normal(0, 0.3, n2)
    y2 = r2 * np.sin(phi2) * np.sin(theta2) + np.random.normal(0, 0.3, n2)
    z2 = r2 * np.cos(phi2) + np.random.normal(0, 0.3, n2)
    points_group2 = np.column_stack([x2, y2, z2])
    
    # 将组2移动到不同位置
    points_group2 += np.array([4, 2, 1])
    
    print(f"生成数据: 组1 {len(points_group1)}个点, 组2 {len(points_group2)}个点")
    
    # 测试不同的分离方法
    methods = {
        'SVM (Linear)': lambda: create_svm_separation_surface(
            points_group1, points_group2, kernel='linear', C=1.0
        ),
        'SVM (RBF)': lambda: create_svm_separation_surface(
            points_group1, points_group2, kernel='rbf', C=1.0
        ),
        'SVM (Poly)': lambda: create_svm_separation_surface(
            points_group1, points_group2, kernel='poly', C=1.0
        ),
        'LDA': lambda: create_lda_separation_surface(points_group1, points_group2),
        'K-means': lambda: create_kmeans_separation_surface(points_group1, points_group2)
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n正在测试 {method_name}...")
        try:
            if 'SVM' in method_name:
                surface, model, info = method_func()
                results[method_name] = {
                    'surface': surface,
                    'model': model,
                    'info': info,
                    'type': 'svm'
                }
                print(f"  - 准确率: {info['accuracy']:.3f}")
                print(f"  - 分离度: {info['separation_degree']:.3f}")
                print(f"  - 支持向量数: {info['support_vectors_count']}")
            else:
                surface, model = method_func()
                results[method_name] = {
                    'surface': surface,
                    'model': model,
                    'type': 'other'
                }
                print(f"  - 完成")
        except Exception as e:
            print(f"  - 错误: {e}")
            results[method_name] = None
    
    # 可视化所有方法的结果
    _visualize_all_separation_methods(points_group1, points_group2, results)
    
    return results

def _visualize_all_separation_methods(points_group1, points_group2, results):
    """
    可视化所有分离方法的结果对比
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 计算子图布局
    n_methods = len([r for r in results.values() if r is not None])
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    fig.suptitle('不同分离方法效果对比', fontsize=16)
    
    plot_idx = 1
    
    for method_name, result in results.items():
        if result is None:
            continue
            
        ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
        
        # 绘制原始点云
        ax.scatter(points_group1[:, 0], points_group1[:, 1], points_group1[:, 2], 
                  c='red', s=20, alpha=0.7, label='组1')
        ax.scatter(points_group2[:, 0], points_group2[:, 1], points_group2[:, 2], 
                  c='blue', s=20, alpha=0.7, label='组2')
        
        # 绘制分离曲面
        surface = result['surface']
        ax.plot_surface(surface[:, :, 0], surface[:, :, 1], surface[:, :, 2], 
                       alpha=0.3, color='green')
        
        ax.set_title(f'{method_name}')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        
        # 添加统计信息
        if result['type'] == 'svm':
            info = result['info']
            stats_text = f"准确率: {info['accuracy']:.3f}\n分离度: {info['separation_degree']:.3f}"
            ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plot_idx += 1
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = "separation_visualization"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "separation_methods_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n分离方法对比图已保存: {save_path}")
    
    # 打印总结
    print("\n=== 分离方法总结 ===")
    for method_name, result in results.items():
        if result is not None and result['type'] == 'svm':
            info = result['info']
            print(f"{method_name}:")
            print(f"  - 准确率: {info['accuracy']:.3f}")
            print(f"  - 分离度: {info['separation_degree']:.3f}")
            print(f"  - 支持向量数: {info['support_vectors_count']}")
            print()

def compare_separation_methods_for_vascular_data(tree_json: str):
    """
    针对血管数据比较不同分离方法
    """
    print("=== 血管数据分离方法比较 ===")
    
    # 加载血管数据
    with open(tree_json, 'r') as fp:
        tree_data = json.load(fp)
    
    # 提取主干和分支点
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(tree_data)
    
    print(f"主干点: {len(trunk_pts)}")
    print(f"分支1点: {len(br1_pts)}")
    print(f"分支2点: {len(br2_pts)}")
    
    # 比较不同方法分离主干和分支
    methods = {
        'SVM (Linear)': lambda: create_svm_separation_surface(
            trunk_pts, np.vstack([br1_pts, br2_pts]), kernel='linear', C=1.0
        ),
        'SVM (RBF)': lambda: create_svm_separation_surface(
            trunk_pts, np.vstack([br1_pts, br2_pts]), kernel='rbf', C=1.0
        ),
        'LDA': lambda: create_lda_separation_surface(
            trunk_pts, np.vstack([br1_pts, br2_pts])
        ),
        'K-means': lambda: create_kmeans_separation_surface(
            trunk_pts, np.vstack([br1_pts, br2_pts])
        )
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n正在测试 {method_name}...")
        try:
            if 'SVM' in method_name:
                surface, model, info = method_func()
                results[method_name] = {
                    'surface': surface,
                    'model': model,
                    'info': info,
                    'type': 'svm'
                }
                print(f"  - 准确率: {info['accuracy']:.3f}")
                print(f"  - 分离度: {info['separation_degree']:.3f}")
            else:
                surface, model = method_func()
                results[method_name] = {
                    'surface': surface,
                    'model': model,
                    'type': 'other'
                }
                print(f"  - 完成")
        except Exception as e:
            print(f"  - 错误: {e}")
            results[method_name] = None
    
    # 可视化血管分离结果
    _visualize_vascular_separation(trunk_pts, br1_pts, br2_pts, results, tree_json)
    
    return results

def _visualize_vascular_separation(trunk_pts, br1_pts, br2_pts, results, tree_json):
    """
    可视化血管分离结果
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 计算子图布局
    n_methods = len([r for r in results.values() if r is not None])
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(12, 6 * n_rows))
    fig.suptitle('血管数据分离方法比较', fontsize=16)
    
    plot_idx = 1
    
    for method_name, result in results.items():
        if result is None:
            continue
            
        ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
        
        # 绘制血管点云
        ax.scatter(trunk_pts[:, 0], trunk_pts[:, 1], trunk_pts[:, 2], 
                  c='red', s=20, alpha=0.7, label='主干')
        ax.scatter(br1_pts[:, 0], br1_pts[:, 1], br1_pts[:, 2], 
                  c='blue', s=20, alpha=0.7, label='分支1')
        ax.scatter(br2_pts[:, 0], br2_pts[:, 1], br2_pts[:, 2], 
                  c='green', s=20, alpha=0.7, label='分支2')
        
        # 绘制分离曲面
        surface = result['surface']
        ax.plot_surface(surface[:, :, 0], surface[:, :, 1], surface[:, :, 2], 
                       alpha=0.3, color='yellow')
        
        ax.set_title(f'{method_name}')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        
        # 添加统计信息
        if result['type'] == 'svm':
            info = result['info']
            stats_text = f"准确率: {info['accuracy']:.3f}\n分离度: {info['separation_degree']:.3f}"
            ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plot_idx += 1
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = "separation_visualization"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vascular_separation_{os.path.basename(tree_json)}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n血管分离结果已保存: {save_path}")

# --------- Dataset ---------