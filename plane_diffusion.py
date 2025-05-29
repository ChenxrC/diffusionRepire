import json
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
from visual import compute_tree_plane_normals

# ------------------------
# 数据预处理
# ------------------------
try:
    find_max_points_branches  # type: ignore
except NameError:
    def find_max_points_branches(tree_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """遍历 tree_data 取得主干与点数最多的两条一级分支点云
        返回 (trunk_pts, branch1_pts, branch2_pts)
        若一级分支不足 2 条则返回空数组占位"""
        trunk_points = np.array(tree_data["branches"][0]["points"], dtype=np.float32)
        branches = tree_data["branches"][0].get("children", [])
        if len(branches) < 2:
            # fallback: 直接按 children 全部或自己复制
            if len(branches) == 1:
                branch1_pts = np.array(branches[0]["points"], dtype=np.float32)
                branch2_pts = branch1_pts.copy()
            else:
                branch1_pts = branch2_pts = trunk_points.copy()
            return trunk_points, branch1_pts, branch2_pts
        # sort by number of points
        branches_sorted = sorted(branches, key=lambda b: len(b["points"]), reverse=True)
        branch1_pts = np.array(branches_sorted[0]["points"], dtype=np.float32)
        branch2_pts = np.array(branches_sorted[1]["points"], dtype=np.float32)
        return trunk_points, branch1_pts, branch2_pts

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
def compute_tree_plane_normals(tree_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """根据树形数据计算分支平面与主干平面的法向量
    
    Args:
        tree_data: 树形 JSON 数据 (包含 branches 列表)
    Returns:
        (normal_branch, normal_trunk) : 两个单位法向量
    """
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(tree_data)
    branch_pts = np.vstack([br1_pts, br2_pts])
    # 分支平面法向量
    normal_branch, _ = calculate_plane_equation(branch_pts)
    normal_branch = normal_branch / np.linalg.norm(normal_branch)
    # 主干平面法向量 (要求与 branch 平面垂直)
    normal_trunk, _ = calculate_perpendicular_plane(trunk_pts, normal_branch)
    normal_trunk = normal_trunk / np.linalg.norm(normal_trunk)
    return normal_branch, normal_trunk

def generate_noisy_normals(base_normal: np.ndarray, n: int, max_angle_deg: float = 20.0, seed: int = None) -> np.ndarray:
    """围绕基准法向量生成带有角度扰动的法向量
    
    Args:
        base_normal : 基准单位法向量 (3,)
        n           : 生成数量
        max_angle_deg : 在每个欧拉角方向的最大偏转角度(度)
        seed        : 随机种子 (可选)
    Returns:
        noisy_normals : (n,3) 的单位向量数组
    """
    if seed is not None:
        np.random.seed(seed)
    base_normal = base_normal / np.linalg.norm(base_normal)
    noisy_list = []
    for _ in range(n):
        # 在 (-max_angle, max_angle) 范围随机采样三个欧拉角
        angles = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg, size=3))
        rx, ry, rz = angles
        # 分别构建绕 x/y/z 轴的小角度旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(rx), -np.sin(rx)],
                        [0, np.sin(rx),  np.cos(rx)]])
        R_y = np.array([[ np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
        R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                        [np.sin(rz),  np.cos(rz), 0],
                        [0, 0, 1]])
        # 组合旋转 (Z * Y * X)
        R = R_z @ R_y @ R_x
        new_normal = R @ base_normal
        new_normal /= np.linalg.norm(new_normal)
        noisy_list.append(new_normal)
    return np.array(noisy_list)
if __name__=='__main__':
    # 假设当前目录有多个 tree json
    import glob, os
    files = glob.glob('tree_*.json')
    if len(files)>=1:
        model = train_tree_plane_predictor(files, epochs=50, device='cpu')
        test_normal = predict_plane_normal(files[0], model)
        print('Pred normal:', test_normal)
