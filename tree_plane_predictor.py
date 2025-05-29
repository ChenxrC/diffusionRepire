import json
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from visual import compute_tree_plane_normals

# ------------------------
# 数据预处理
# ------------------------

def tree_points_to_array(tree_data: Dict[str, Any]) -> np.ndarray:
    """递归遍历树, 返回 (N,5): x,y,z,branch_id,parent_id"""
    pts = []
    def dfs(branch: Dict[str,Any], parent_id: int):
        bid = branch.get('id', 0)  # 假设已在 JSON 中标注 id
        for p in branch['points']:
            pts.append([p[0], p[1], p[2], bid, parent_id])
        for child in branch.get('children', []):
            dfs(child, bid)
    # 假设根在 branches[0]
    root = tree_data['branches'][0]
    dfs(root, 0)
    return np.array(pts, dtype=np.float32)

# ------------------------
# 数据集
# ------------------------
class TreePlaneDataset(Dataset):
    def __init__(self, json_files: List[str]):
        self.files = json_files
        self.all_data = []
        self.targets = []
        for f in json_files:
            with open(f,'r') as fp:
                td = json.load(fp)
            pos = tree_points_to_array(td)   # (N,5)
            n_branch, _ = compute_tree_plane_normals(td)
            self.all_data.append(pos)
            self.targets.append(n_branch)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        pts = self.all_data[idx]
        target = self.targets[idx]
        # 归一化坐标
        xyz = pts[:,:3]
        xyz = xyz - xyz.mean(0, keepdims=True)
        xyz /= (xyz.std()+1e-6)
        # branch与parent id 归一化
        ids = pts[:,3:5] / 100.0
        feat = np.concatenate([xyz, ids], axis=1) # (N,5)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# ------------------------
# PointNet-like Encoder
# ------------------------
class PointEncoder(nn.Module):
    def __init__(self, feat_dim=5, emb_dim=128):
        super().__init__()
        self.mlp1 = nn.Linear(feat_dim, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, emb_dim)
        self.head = nn.Linear(emb_dim, 3)
    def forward(self, pts: torch.Tensor):
        # pts: (B,N,F)
        x = F.relu(self.mlp1(pts))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = x.max(dim=1)[0]  # 全局汇聚
        normal_pred = self.head(x)
        normal_pred = F.normalize(normal_pred, dim=1)  # 单位向量
        return normal_pred

# ------------------------
# 训练函数
# ------------------------

def train_tree_plane_predictor(train_files: List[str], epochs:int=100, batch_size:int=4, lr:float=1e-3, device='cpu'):
    ds = TreePlaneDataset(train_files)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = PointEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        total=0; cnt=0
        for feats,target in dl:
            feats,target = feats.to(device), target.to(device)
            pred = model(feats)
            # MSE/ cosine loss: maximize cos similarity
            loss = 1 - (pred*target).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); cnt+=1
        if ep%10==0:
            print(f"Epoch {ep}/{epochs}  loss {total/cnt:.4f}")
    return model

# ------------------------
# 推理函数
# ------------------------

def predict_plane_normal(tree_json: str, model: PointEncoder, device='cpu')->np.ndarray:
    with open(tree_json,'r') as fp:
        td = json.load(fp)
    pts = tree_points_to_array(td)
    xyz = pts[:,:3]
    xyz = xyz - xyz.mean(0, keepdims=True)
    xyz /= (xyz.std()+1e-6)
    ids = pts[:,3:5]/100.0
    feat = np.concatenate([xyz, ids], axis=1)
    with torch.no_grad():
        pred = model(torch.tensor(feat[None,...],dtype=torch.float32,device=device))
    return pred.squeeze().cpu().numpy()

if __name__=='__main__':
    # 假设当前目录有多个 tree json
    import glob, os
    files = glob.glob('tree_*.json')
    if len(files)>=1:
        model = train_tree_plane_predictor(files, epochs=50, device='cpu')
        test_normal = predict_plane_normal(files[0], model)
        print('Pred normal:', test_normal) 