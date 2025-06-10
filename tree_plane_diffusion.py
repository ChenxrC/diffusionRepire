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
            
            # 计算分支平面的法向量和中心点
            n_branch,_ = compute_tree_plane_normals(td)
            trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
            branch_pts = np.vstack([br1_pts, br2_pts])
            branch_center = branch_pts.mean(axis=0)
            
            # 生成平面上的均匀分布点
            plane_points = generate_plane_points(branch_center, n_branch, plane_size=30.0, grid_size=grid_size)
            self.targets.append(plane_points.astype(np.float32))
            
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