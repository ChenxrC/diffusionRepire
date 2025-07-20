"""
血管树扩散数据集
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List

from ..core.surface_generation import generate_surface_grid, calculate_centerline_from_branches
from ..utils.vascular_utils import find_max_points_branches
from ..utils.cascade_transform import apply_cascade_transform_to_branches


class TreeNormalDiffusionDataset(Dataset):
    """
    血管树扩散数据集
    """
    
    def __init__(self, json_files: List[str], grid_size: int = 32, point_spacing: float = 0.2):
        """
        初始化数据集
        
        Args:
            json_files: JSON文件路径列表
            grid_size: 网格大小
            point_spacing: 点间距
        """
        self.files = json_files
        self.grid_size = grid_size
        self.point_spacing = point_spacing
        self.data = []
        self.targets = []
        
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """加载和处理数据"""
        for f in self.files:
            with open(f, 'r') as fp:
                td = json.load(fp)
            
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
                self._process_single_tree(td_aug)
    
    def _process_single_tree(self, tree_data: dict):
        """处理单个血管树数据"""
        from ..utils.tree_utils import tree_points_to_array
        
        pts = tree_points_to_array(tree_data)
        self.data.append(pts)
        
        # 获取主干和分支点
        trunk_pts, br1_pts, br2_pts = find_max_points_branches(tree_data)
        
        # 计算中轴线
        centerline_points, main_direction = calculate_centerline_from_branches(
            trunk_pts, br1_pts, br2_pts
        )
        
        # 生成曲面上的网格点
        surface_grid_points = generate_surface_grid(
            centerline_points, main_direction, self.grid_size, self.point_spacing
        )
        self.targets.append(surface_grid_points.astype(np.float32))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            feat: 特征张量
            target: 目标张量
        """
        pts = self.data[idx]
        target = self.targets[idx]  # (grid_size, grid_size, 3)
        
        # 标准化xyz坐标
        xyz = pts[:, :3]
        xyz = xyz - xyz.mean(0, keepdims=True)
        xyz = xyz / (xyz.std() + 1e-6)
        
        # 处理ID特征
        ids = pts[:, 3:5] / 100.0
        
        # 组合特征
        feat = np.concatenate([xyz, ids], axis=1)
        
        # 将目标展平为一维向量
        target_flat = target.flatten()  # (grid_size*grid_size*3,)
        
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target_flat, dtype=torch.float32) 