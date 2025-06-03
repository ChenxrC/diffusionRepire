import json
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pdb
from visual import compute_tree_plane_normals
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
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# --------- Conditioned Noise Predictor ---------
class CondNoisePredictor(nn.Module):
    def __init__(self, feat_dim=5, emb_dim=128):
        super().__init__()
        self.encoder = PointEncoder(feat_dim, emb_dim)
        self.time_fc = nn.Linear(1, emb_dim)
        self.cond_fc = nn.Linear(3, emb_dim)  # 将 PointEncoder 输出映射到 emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+3, 128),
            nn.SiLU(),
            nn.Linear(128, 3)
        )
    def forward(self, pts:torch.Tensor, noisy_n:torch.Tensor, t:torch.Tensor):
        # pts: (B,N,F)  noisy_n:(B,3) t:(B,)
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
    x = F.normalize(torch.randn(1,3,device=device), dim=1)
    with torch.no_grad():
        for t_inv in range(T-1,-1,-1):
            t=torch.tensor([t_inv],device=device)
            beta=betas[t_inv]
            pred_noise = model(feats, x, t)
            x = (x - torch.sqrt(beta)*pred_noise)/torch.sqrt(1-beta)
            x = F.normalize(x,dim=1)
    return x.squeeze().cpu().numpy()

if __name__=='__main__':
    import glob
    files = glob.glob('tree_*.json')
    if len(files):
        model, betas = train_tree_diffusion(files, epochs=500, device='cpu')
        pred = denoise_with_tree(files[0], model, betas)
        print('Pred normal diff:', pred) 