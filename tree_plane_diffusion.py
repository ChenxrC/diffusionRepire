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
    def __init__(self, json_files: List[str], grid_size=32):
        self.files = json_files
        self.grid_size = grid_size
        self.data = []
        self.targets = []
        for f in json_files:
            with open(f,'r') as fp:
                td=json.load(fp)
            pts = tree_points_to_array(td)
            self.data.append(pts)
            
            # 获取主干和分支点
            trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
            
            # 计算两个分支对应点的中点
            min_len = min(len(br1_pts), len(br2_pts))
            if min_len > 0:
                # 取相同数量的点来计算中点
                br1_sampled = br1_pts[:min_len]
                br2_sampled = br2_pts[:min_len]
                midpoints = (br1_sampled + br2_sampled) / 2.0
            else:
                # 如果没有分支点，使用主干末端作为中点
                midpoints = np.array([trunk_pts[-1]])
            
            # 合并主干点和中点来拟合曲面
            surface_points = np.vstack([trunk_pts, midpoints])
            
            # 使用PCA找到主要方向
            centroid = surface_points.mean(axis=0)
            centered_points = surface_points - centroid
            
            # 计算协方差矩阵和主成分
            cov_matrix = np.cov(centered_points.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # 按特征值排序（降序）
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # 主方向和次方向
            primary_dir = eigenvecs[:, 0]    # 最大特征值对应的方向
            secondary_dir = eigenvecs[:, 1]  # 第二大特征值对应的方向
            
            # 计算曲面的范围
            proj_primary = np.dot(centered_points, primary_dir)
            proj_secondary = np.dot(centered_points, secondary_dir)
            
            primary_range = [proj_primary.min(), proj_primary.max()]
            secondary_range = [proj_secondary.min(), proj_secondary.max()]
            
            # 在主平面上生成32x32网格点
            u = np.linspace(primary_range[0], primary_range[1], grid_size)
            v = np.linspace(secondary_range[0], secondary_range[1], grid_size)
            U, V = np.meshgrid(u, v)
            
            # 将网格点转换回3D空间
            grid_points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    # 在主平面上的点
                    point_2d = U[i,j] * primary_dir + V[i,j] * secondary_dir
                    
                    # 添加第三维的弯曲（基于到中心的距离）
                    dist_from_center = np.sqrt(U[i,j]**2 + V[i,j]**2)
                    curvature = 0.1 * dist_from_center * np.sin(dist_from_center * 0.1)  # 轻微弯曲
                    
                    # 第三个方向（法向量）
                    normal_dir = eigenvecs[:, 2]
                    point_3d = centroid + point_2d + curvature * normal_dir
                    
                    grid_points.append(point_3d)
            
            surface_grid_points = np.array(grid_points)
            self.targets.append(surface_grid_points.astype(np.float32))
            
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        pts = self.data[idx]
        target = self.targets[idx]  # (grid_size*grid_size, 3)
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

def train_tree_diffusion(train_files: List[str], T:int=100, epochs:int=10000, batch_size:int=2, device='cpu', grid_size=32):
    dataset = TreeNormalDiffusionDataset(train_files, grid_size=grid_size)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CondNoisePredictor(grid_size=grid_size).to(device)
    betas = linear_beta_schedule(T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for ep in range(1, epochs+1):
        total=0;cnt=0
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
            total+=loss.item(); cnt+=1
        if ep%100==0:
            print(f"Epoch {ep}/{epochs} loss {total/cnt:.6f}")
    return model, betas

# --------- Denoise ---------

def denoise_with_tree(tree_json:str, model:CondNoisePredictor, betas:torch.Tensor, device='cpu', grid_size=32):
    with open(tree_json,'r') as fp:
        td=json.load(fp)
    pts_arr = tree_points_to_array(td)
    xyz = pts_arr[:,:3]
    xyz = xyz-xyz.mean(0,keepdims=True)
    xyz = xyz/(xyz.std()+1e-6)
    ids = pts_arr[:,3:5]/100.0
    feats = torch.tensor(np.concatenate([xyz,ids],axis=1)[None,...],dtype=torch.float32,device=device)
    T = betas.shape[0]
    
    # 初始化为随机分布的点云
    x = torch.randn(1, grid_size * grid_size * 3, device=device)
    
    with torch.no_grad():
        for t_inv in range(T-1,-1,-1):
            t=torch.tensor([t_inv],device=device)
            beta=betas[t_inv]
            pred_noise = model(feats, x, t)
            x = (x - torch.sqrt(beta)*pred_noise)/torch.sqrt(1-beta)
    return x.squeeze().cpu().numpy().reshape(grid_size, grid_size, 3)

# --------- Denoise with GIF ---------

def denoise_with_gif(tree_json:str, model:CondNoisePredictor, betas:torch.Tensor, gif_path:str='denoise.gif', device='cpu', grid_size=32):
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
    
    # 初始化为随机分布的点云
    x = torch.randn(1, grid_size * grid_size * 3, device=device)
    
    frames=[]
    # precompute point sets for scatter
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
    branch_pts = np.vstack([br1_pts, br2_pts])

    with torch.no_grad():
        for t_inv in range(T-1,-1,-1):
            # 将当前点云重塑为(grid_size, grid_size, 3)格式
            current_points = x.squeeze().cpu().numpy().reshape(grid_size, grid_size, 3)
            
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制原始血管点
            ax.scatter(*trunk_pts.T,  c='blue',  s=2, alpha=0.6)
            ax.scatter(*br1_pts.T,    c='green', s=2, alpha=0.6)
            ax.scatter(*br2_pts.T,    c='red',   s=2, alpha=0.6)
            
            # 绘制当前预测的点云
            points_flat = current_points.reshape(-1, 3)
            ax.scatter(*points_flat.T, c='orange', s=1, alpha=0.8)
            
            # 绘制点云的网格线以显示结构
            for i in range(0, grid_size, 4):  # 每4行绘制一条线
                line_points = current_points[i, :, :]
                ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                       color='red', alpha=0.3, linewidth=0.5)
            for j in range(0, grid_size, 4):  # 每4列绘制一条线
                line_points = current_points[:, j, :]
                ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
                       color='red', alpha=0.3, linewidth=0.5)

            # 设置视图
            cen_branch = branch_pts.mean(axis=0)
            ax.set_xlim(cen_branch[0]-40, cen_branch[0]+40)
            ax.set_ylim(cen_branch[1]-40, cen_branch[1]+40)
            ax.set_zlim(cen_branch[2]-40, cen_branch[2]+40)
            ax.set_title(f'Denoising Step: {T-t_inv-1}/{T}')
            ax.set_axis_off()
            
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

def visualize_generated_surface(tree_json: str, generated_points: np.ndarray, save_path: str = None, show_wireframe: bool = True):
    """
    可视化生成的曲面与原始血管树的对比
    
    Args:
        tree_json: 血管树json文件路径
        generated_points: 生成的曲面点，形状为(grid_size, grid_size, 3)
        save_path: 保存图片的路径，如果为None则显示
        show_wireframe: 是否显示网格线
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    else:
        plt.show()
    
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

def comprehensive_surface_validation(tree_json: str, generated_points: np.ndarray, save_prefix: str = "surface_validation"):
    """
    综合曲面验证：可视化 + 质量分析
    
    Args:
        tree_json: 血管树json文件路径
        generated_points: 生成的曲面点，形状为(grid_size, grid_size, 3)
        save_prefix: 保存文件的前缀
    """
    print("开始综合曲面验证...")
    
    # 1. 可视化
    print("1. 生成可视化图像...")
    visualize_generated_surface(tree_json, generated_points, f"{save_prefix}_visualization.png")
    
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
    
    return metrics

if __name__=='__main__':
    import glob
    files = glob.glob('tree_*.json')
    if len(files):
        grid_size = 32
        model, betas = train_tree_diffusion(files, epochs=50000, device='cpu', grid_size=grid_size)
        pred = denoise_with_tree(files[0], model, betas, grid_size=grid_size)
        pred_gif = denoise_with_gif(files[0], model, betas, gif_path='denoise.gif', grid_size=grid_size)
        print('Predicted plane points shape:', pred.shape)  # (32, 32, 3)
        print('Final points center:', pred.mean(axis=(0,1))) 
        
        # 新增：综合验证生成的曲面
        print("\n开始验证生成的曲面...")
        comprehensive_surface_validation(files[0], pred, "trained_surface") 