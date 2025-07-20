"""
树数据工具函数
"""

import numpy as np
from typing import Dict, Any


def tree_points_to_array(tree_data: Dict[str, Any]) -> np.ndarray:
    """
    将血管树数据转换为点云数组
    
    Args:
        tree_data: 血管树数据字典
    
    Returns:
        points: 点云数组 (N, 5) - [x, y, z, id1, id2]
    """
    # 这里应该实现具体的数据转换逻辑
    # 暂时返回空数组，需要根据实际数据结构实现
    points = np.array([])
    
    return points 