"""
曲面生成核心功能
"""

import numpy as np
from typing import Tuple


def generate_plane_points(center: np.ndarray, normal: np.ndarray, plane_size: float = 30.0, grid_size: int = 32):
    """
    在给定平面上生成均匀分布的点
    
    Args:
        center: 平面中心点
        normal: 平面法向量
        plane_size: 平面大小
        grid_size: 网格大小
    
    Returns:
        points: 平面上的点 (grid_size*grid_size, 3)
    """
    # 生成平面的两个正交基向量
    helper = np.array([1., 0., 0.])
    if np.allclose(abs(np.dot(helper, normal)), 1.0, atol=1e-3):
        helper = np.array([0., 1., 0.])
    v1 = np.cross(normal, helper)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2)
    
    # 生成网格点
    g = np.linspace(-plane_size/2, plane_size/2, grid_size)
    u, v = np.meshgrid(g, g)
    
    # 计算平面上的点
    points = center + u[..., None]*v1 + v[..., None]*v2
    return points.reshape(-1, 3)  # (grid_size*grid_size, 3)


def generate_surface_grid(centerline_points: np.ndarray, main_direction: np.ndarray, 
                         grid_size: int = 32, point_spacing: float = 0.2) -> np.ndarray:
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
        axis_pos = interpolate_on_centerline(centerline_points, t)
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


def interpolate_on_centerline(centerline_points: np.ndarray, t: float) -> np.ndarray:
    """
    在中轴线上按参数t插值 (t在0到1之间)
    
    Args:
        centerline_points: 中轴线点
        t: 插值参数 (0-1)
    
    Returns:
        interpolated_point: 插值点
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


def calculate_centerline_from_branches(trunk_pts: np.ndarray, br1_pts: np.ndarray, 
                                     br2_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从主干和分支点计算中轴线
    
    Args:
        trunk_pts: 主干点
        br1_pts: 分支1点
        br2_pts: 分支2点
    
    Returns:
        centerline_points: 中轴线点
        main_direction: 主方向
    """
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
    
    return sorted_centerline, main_direction 