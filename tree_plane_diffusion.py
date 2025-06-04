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

# --------- Dataset ---------
class TreeNormalDiffusionDataset(Dataset):
    def __init__(self, json_files: List[str]):
        self.files = json_files
        self.data = []
        self.targets = []
        for f in json_files:
            with open(f,'r') as fp:
                td=json.load(fp)
            pts = tree_points_to_array(td)
            self.data.append(pts)
            n_branch,_ = compute_tree_plane_normals(td)
            self.targets.append(n_branch.astype(np.float32))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        pts = self.data[idx]
        target = self.targets[idx]
        # normalize xyz
        xyz = pts[:,:3]
        xyz = xyz - xyz.mean(0, keepdims=True)
        xyz = xyz/(xyz.std()+1e-6)
        ids = pts[:,3:5]/100.0
        feat = np.concatenate([xyz, ids], axis=1)
        # 修改目标数据以包含法向量和连接点坐标
        # pdb.set_trace()
        target = np.concatenate([target, pts[0, :3]], axis=0)  # 假设连接点坐标在 pts 的前 3 列
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# --------- Conditioned Noise Predictor ---------
class CondNoisePredictor(nn.Module):
    def __init__(self, feat_dim=5, emb_dim=128):
        super().__init__()
        self.encoder = PointEncoder(feat_dim, emb_dim)
        self.time_fc = nn.Linear(1, emb_dim)
        self.cond_fc = nn.Linear(3, emb_dim)  # 将 PointEncoder 输出映射到 emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+6, 128),
            nn.SiLU(),
            nn.Linear(128, 6)  # 修改输出维度为 6
        )
    def forward(self, pts:torch.Tensor, noisy_n:torch.Tensor, t:torch.Tensor):
        # pts: (B,N,F)  noisy_n:(B,6) t:(B,)
        cond_raw = self.encoder(pts)           # (B,3)
        cond_emb = self.cond_fc(cond_raw)      # (B,emb_dim)
        time_emb = torch.sin(self.time_fc(t.float().unsqueeze(1)))  # (B,emb_dim)
        h = torch.cat([cond_emb + time_emb, noisy_n], dim=1)
        return self.mlp(h)

# --------- Training function ---------

def train_tree_diffusion(train_files: List[str], T:int=100, epochs:int=10000, batch_size:int=2, device='cpu'):
    dataset = TreeNormalDiffusionDataset(train_files)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CondNoisePredictor().to(device)
    betas = linear_beta_schedule(T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs+1):
        total=0;cnt=0
        for feats, clean in dl:
            feats, clean = feats.to(device), clean.to(device)
            B = clean.shape[0]
            t = torch.randint(0, T, (B,), device=device)
            beta_t = betas[t].unsqueeze(1)
            noise = torch.randn_like(clean)
            noisy = torch.sqrt(1-beta_t)*clean + torch.sqrt(beta_t)*noise
            noisy = F.normalize(noisy, dim=1)
            pred_noise = model(feats, noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item(); cnt+=1
        if ep%100==0:
            print(f"Epoch {ep}/{epochs} loss {total/cnt:.6f}")
    return model, betas

# --------- Denoise ---------

def denoise_with_tree(tree_json:str, model:CondNoisePredictor, betas:torch.Tensor, device='cpu'):
    with open(tree_json,'r') as fp:
        td=json.load(fp)
    pts_arr = tree_points_to_array(td)
    xyz = pts_arr[:,:3]
    xyz = xyz-xyz.mean(0,keepdims=True)
    xyz = xyz/(xyz.std()+1e-6)
    ids = pts_arr[:,3:5]/100.0
    feats = torch.tensor(np.concatenate([xyz,ids],axis=1)[None,...],dtype=torch.float32,device=device)
    T = betas.shape[0]
    x = F.normalize(torch.randn(1,6,device=device), dim=1)  # 修改为 6 维
    with torch.no_grad():
        for t_inv in range(T-1,-1,-1):
            t=torch.tensor([t_inv],device=device)
            beta=betas[t_inv]
            pred_noise = model(feats, x, t)
            x = (x - torch.sqrt(beta)*pred_noise)/torch.sqrt(1-beta)
            x = F.normalize(x,dim=1)
    return x.squeeze().cpu().numpy()

# --------- Denoise with GIF ---------

def denoise_with_gif(tree_json:str, model:CondNoisePredictor, betas:torch.Tensor, gif_path:str='denoise.gif', device='cpu', max_angle_deg:int=80):
    """与 denoise_with_tree 相同，但每一步将法向量渲染为箭头并保存 GIF"""
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
    # 选取与目标平面有较大偏差的初始法向量
    branch_gt,_ = compute_tree_plane_normals(td)
    noisy_init = generate_noisy_normals(branch_gt,1,max_angle_deg=max_angle_deg,seed=None)[0]
    x = torch.tensor(noisy_init, dtype=torch.float32, device=device).unsqueeze(0)
    x = F.normalize(x, dim=1)
    frames=[]
    # precompute point sets for scatter
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
    branch_pts = np.vstack([br1_pts, br2_pts])

    def plane_mesh(center, normal, plane_size=30.0, density=15):
        helper = np.array([1.,0.,0.])
        if np.allclose(abs(np.dot(helper, normal)),1.0,atol=1e-3):
            helper=np.array([0.,1.,0.])
        v1=np.cross(normal,helper); v1/=np.linalg.norm(v1)
        v2=np.cross(normal,v1); v2/=np.linalg.norm(v2)
        g=np.linspace(-plane_size/2,plane_size/2,density)
        u,v=np.meshgrid(g,g)
        pts=center+u[...,None]*v1+v[...,None]*v2
        return pts[...,0],pts[...,1],pts[...,2]

    with torch.no_grad():
        for t_inv in range(T-1,-1,-1):
            # render current state
            n_vec = x.squeeze().cpu().numpy(); n_vec/=np.linalg.norm(n_vec)
            n_perp, _ = calculate_perpendicular_plane(trunk_pts, n_vec); n_perp/=np.linalg.norm(n_perp)

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*trunk_pts.T,  c='blue',  s=5)
            ax.scatter(*br1_pts.T,    c='green', s=5)
            ax.scatter(*br2_pts.T,    c='red',   s=5)

            cen_branch = branch_pts.mean(axis=0)
            cen_trunk  = trunk_pts.mean(axis=0)
            X1,Y1,Z1 = plane_mesh(cen_branch, n_vec)
            X2,Y2,Z2 = plane_mesh(cen_trunk,  n_perp)
            ax.plot_surface(X1,Y1,Z1,alpha=0.2,color='lightgreen')
            ax.plot_surface(X2,Y2,Z2,alpha=0.2,color='lightblue')
            ax.quiver(*cen_branch, *n_vec, length=5,color='green')
            ax.quiver(*cen_trunk,  *n_perp,length=5,color='blue')

            ax.set_xlim(cen_branch[0]-30, cen_branch[0]+30)
            ax.set_ylim(cen_branch[1]-30, cen_branch[1]+30)
            ax.set_zlim(cen_branch[2]-30, cen_branch[2]+30)
            ax.set_axis_off()
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            frames.append(frame)
            plt.close(fig)

            # diffusion step
            t=torch.tensor([t_inv],device=device)
            beta=betas[t_inv]
            pred_noise = model(feats, x, t)
            x = (x - torch.sqrt(beta)*pred_noise)/torch.sqrt(1-beta)
            x = F.normalize(x,dim=1)
    imageio.mimsave(gif_path, frames, fps=10)
    return x.squeeze().cpu().numpy()

if __name__=='__main__':
    import glob
    files = glob.glob('tree_*.json')
    if len(files):
        model, betas = train_tree_diffusion(files, epochs=10000, device='cpu')
        pred = denoise_with_tree(files[0], model, betas)
        pred = denoise_with_gif(files[0], model, betas, gif_path='denoise.gif')
        print('Pred normal diff:', pred) 