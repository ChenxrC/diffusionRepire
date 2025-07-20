#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascade Transform Functions
级联变换函数 - 用于增强血管形状多样性

功能：
- 级联偏移：前序节点偏移时，后续节点跟随偏移
- 级联旋转：前序节点旋转时，后续节点跟随旋转
- 保持连接性：确保节点间距离不超过1
- 形状多样性增强
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json


class CascadeTransformer:
    """级联变换器类"""
    
    def __init__(self, max_offset_distance=1.0, rotation_angle_range=15.0):
        """
        初始化级联变换器
        
        Args:
            max_offset_distance: 最大偏移距离
            rotation_angle_range: 旋转角度范围（度）
        """
        self.max_offset_distance = max_offset_distance
        self.rotation_angle_range = np.radians(rotation_angle_range)
    
    def apply_cascade_offset(self, points, branch_hierarchy, offset_strength=0.3):
        """
        应用级联偏移变换
        
        Args:
            points: 所有点的坐标 (N, 3)
            branch_hierarchy: 分支层次结构，格式为 {branch_id: [point_indices]}
            offset_strength: 偏移强度 (0-1)
        
        Returns:
            transformed_points: 变换后的点坐标
        """
        transformed_points = points.copy()
        
        # 为每个分支应用级联偏移
        for branch_id, point_indices in branch_hierarchy.items():
            if len(point_indices) < 2:
                continue
            
            # 生成随机偏移方向
            offset_direction = np.random.randn(3)
            offset_direction = offset_direction / np.linalg.norm(offset_direction)
            
            # 计算偏移量
            offset_magnitude = self.max_offset_distance * offset_strength
            
            # 应用级联偏移
            for i, point_idx in enumerate(point_indices):
                # 偏移量随距离衰减
                decay_factor = np.exp(-i * 0.1)  # 指数衰减
                current_offset = offset_direction * offset_magnitude * decay_factor
                
                # 应用偏移
                transformed_points[point_idx] += current_offset
                
                # 确保与前序节点的距离不超过限制
                if i > 0:
                    prev_idx = point_indices[i-1]
                    current_pos = transformed_points[point_idx]
                    prev_pos = transformed_points[prev_idx]
                    distance = np.linalg.norm(current_pos - prev_pos)
                    
                    if distance > self.max_offset_distance:
                        # 调整位置以保持距离限制
                        direction = (current_pos - prev_pos) / distance
                        transformed_points[point_idx] = prev_pos + direction * self.max_offset_distance
        
        return transformed_points
    
    def apply_cascade_rotation(self, points, branch_hierarchy, rotation_strength=0.5):
        """
        应用级联旋转变换
        
        Args:
            points: 所有点的坐标 (N, 3)
            branch_hierarchy: 分支层次结构
            rotation_strength: 旋转强度 (0-1)
        
        Returns:
            transformed_points: 变换后的点坐标
        """
        transformed_points = points.copy()
        
        # 为每个分支应用级联旋转
        for branch_id, point_indices in branch_hierarchy.items():
            if len(point_indices) < 3:
                continue
            
            # 生成随机旋转轴
            rotation_axis = np.random.randn(3)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # 计算旋转角度
            rotation_angle = self.rotation_angle_range * rotation_strength
            
            # 应用级联旋转
            for i in range(1, len(point_indices)):
                point_idx = point_indices[i]
                
                # 旋转角度随距离衰减
                decay_factor = np.exp(-i * 0.15)
                current_angle = rotation_angle * decay_factor
                
                # 获取前序节点作为旋转中心
                center_idx = point_indices[i-1]
                center = transformed_points[center_idx]
                current_point = transformed_points[point_idx]
                
                # 计算相对于中心的向量
                relative_vector = current_point - center
                
                # 应用旋转
                rotated_vector = self._rotate_vector(relative_vector, rotation_axis, current_angle)
                
                # 更新位置
                transformed_points[point_idx] = center + rotated_vector
                
                # 确保距离限制
                distance = np.linalg.norm(rotated_vector)
                if distance > self.max_offset_distance:
                    normalized_vector = rotated_vector / distance
                    transformed_points[point_idx] = center + normalized_vector * self.max_offset_distance
        
        return transformed_points
    
    def _rotate_vector(self, vector, axis, angle):
        """
        绕指定轴旋转向量
        
        Args:
            vector: 要旋转的向量
            axis: 旋转轴（单位向量）
            angle: 旋转角度（弧度）
        
        Returns:
            rotated_vector: 旋转后的向量
        """
        # 使用罗德里格斯旋转公式
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotated_vector = (vector * cos_angle + 
                         np.cross(axis, vector) * sin_angle + 
                         axis * np.dot(axis, vector) * (1 - cos_angle))
        
        return rotated_vector
    
    def apply_combined_transform(self, points, branch_hierarchy, 
                                offset_strength=0.3, rotation_strength=0.4):
        """
        应用组合变换（偏移+旋转）
        
        Args:
            points: 所有点的坐标
            branch_hierarchy: 分支层次结构
            offset_strength: 偏移强度
            rotation_strength: 旋转强度
        
        Returns:
            transformed_points: 变换后的点坐标
        """
        # 先应用偏移
        points_offset = self.apply_cascade_offset(points, branch_hierarchy, offset_strength)
        
        # 再应用旋转
        points_final = self.apply_cascade_rotation(points_offset, branch_hierarchy, rotation_strength)
        
        return points_final
    
    def generate_branch_hierarchy(self, tree_data):
        """
        从血管树数据生成分支层次结构
        
        Args:
            tree_data: 血管树JSON数据
        
        Returns:
            branch_hierarchy: 分支层次结构 {branch_id: [point_indices]}
        """
        branch_hierarchy = {}
        point_counter = 0
        
        def extract_branch_points(branch, branch_id):
            nonlocal point_counter
            
            if "points" in branch:
                points = np.array(branch["points"])
                point_indices = list(range(point_counter, point_counter + len(points)))
                branch_hierarchy[branch_id] = point_indices
                point_counter += len(points)
            
            if "children" in branch:
                for i, child in enumerate(branch["children"]):
                    child_branch_id = f"{branch_id}_{i}"
                    extract_branch_points(child, child_branch_id)
        
        if "branches" in tree_data and len(tree_data["branches"]) > 0:
            extract_branch_points(tree_data["branches"][0], "trunk")
        
        return branch_hierarchy


def visualize_cascade_transform(original_points, transformed_points, branch_hierarchy, 
                               title="级联变换效果对比"):
    """
    可视化级联变换效果
    
    Args:
        original_points: 原始点坐标
        transformed_points: 变换后的点坐标
        branch_hierarchy: 分支层次结构
        title: 图表标题
    """
    fig = plt.figure(figsize=(20, 8))
    
    # 子图1: 原始形状
    ax1 = fig.add_subplot(141, projection='3d')
    
    # 绘制原始点
    ax1.scatter(*original_points.T, c='blue', s=20, alpha=0.8, label='原始点')
    
    # 绘制分支连接
    for branch_id, point_indices in branch_hierarchy.items():
        if len(point_indices) > 1:
            branch_points = original_points[point_indices]
            ax1.plot(*branch_points.T, 'b-', alpha=0.6, linewidth=2)
    
    ax1.set_title('原始血管形状')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 子图2: 变换后形状
    ax2 = fig.add_subplot(142, projection='3d')
    
    # 绘制变换后的点
    ax2.scatter(*transformed_points.T, c='red', s=20, alpha=0.8, label='变换后点')
    
    # 绘制分支连接
    for branch_id, point_indices in branch_hierarchy.items():
        if len(point_indices) > 1:
            branch_points = transformed_points[point_indices]
            ax2.plot(*branch_points.T, 'r-', alpha=0.6, linewidth=2)
    
    ax2.set_title('变换后血管形状')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 子图3: 对比视图
    ax3 = fig.add_subplot(143, projection='3d')
    
    # 绘制原始点（蓝色）
    ax3.scatter(*original_points.T, c='blue', s=15, alpha=0.6, label='原始点')
    
    # 绘制变换后的点（红色）
    ax3.scatter(*transformed_points.T, c='red', s=15, alpha=0.6, label='变换后点')
    
    # 绘制连接线显示偏移
    for i in range(min(len(original_points), len(transformed_points))):
        ax3.plot([original_points[i, 0], transformed_points[i, 0]],
                [original_points[i, 1], transformed_points[i, 1]],
                [original_points[i, 2], transformed_points[i, 2]],
                'g-', alpha=0.3, linewidth=1)
    
    ax3.set_title('变换对比\n(绿线显示偏移)')
    ax3.legend()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # 子图4: 偏移量统计
    ax4 = fig.add_subplot(144)
    
    # 计算每个点的偏移量
    offsets = np.linalg.norm(transformed_points - original_points, axis=1)
    
    ax4.hist(offsets, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(np.mean(offsets), color='red', linestyle='--', 
                label=f'平均偏移: {np.mean(offsets):.3f}')
    ax4.axvline(np.max(offsets), color='orange', linestyle='--', 
                label=f'最大偏移: {np.max(offsets):.3f}')
    
    ax4.set_xlabel('偏移距离')
    ax4.set_ylabel('点数')
    ax4.set_title('偏移量分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 统一设置视图范围
    all_points = np.vstack([original_points, transformed_points])
    center = all_points.mean(axis=0)
    range_val = max(np.std(all_points, axis=0)) * 3
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*50)
    print("级联变换统计信息")
    print("="*50)
    print(f"总点数: {len(original_points)}")
    print(f"分支数: {len(branch_hierarchy)}")
    print(f"平均偏移距离: {np.mean(offsets):.4f}")
    print(f"最大偏移距离: {np.max(offsets):.4f}")
    print(f"最小偏移距离: {np.min(offsets):.4f}")
    print(f"偏移标准差: {np.std(offsets):.4f}")
    print("="*50)


def demo_cascade_transform():
    """演示级联变换功能"""
    print("🎯 级联变换演示")
    print("="*50)
    
    # 查找血管树文件
    import glob
    json_files = glob.glob('tree_*.json')
    
    if not json_files:
        print("❌ 未找到血管树文件，创建示例数据...")
        # 创建示例血管数据
        tree_data = {
            "branches": [{
                "points": [[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3]],
                "children": [
                    {
                        "points": [[3, 0, 3], [3.5, 1, 3.5], [4, 2, 4]],
                        "children": []
                    },
                    {
                        "points": [[3, 0, 3], [3.5, -1, 3.5], [4, -2, 4]],
                        "children": []
                    }
                ]
            }]
        }
    else:
        print(f"📁 找到血管树文件: {json_files[0]}")
        with open(json_files[0], 'r') as fp:
            tree_data = json.load(fp)
    
    # 创建级联变换器
    transformer = CascadeTransformer(max_offset_distance=1.0, rotation_angle_range=20.0)
    
    # 生成分支层次结构
    branch_hierarchy = transformer.generate_branch_hierarchy(tree_data)
    print(f"生成分支层次结构: {len(branch_hierarchy)} 个分支")
    
    # 提取所有点
    all_points = []
    for branch_id, point_indices in branch_hierarchy.items():
        if "points" in tree_data["branches"][0]:
            all_points.extend(tree_data["branches"][0]["points"])
            for child in tree_data["branches"][0].get("children", []):
                if "points" in child:
                    all_points.extend(child["points"])
    
    original_points = np.array(all_points)
    print(f"提取到 {len(original_points)} 个点")
    
    # 应用不同的变换
    print("\n🔄 应用级联变换...")
    
    # 1. 仅偏移变换
    print("1. 应用级联偏移...")
    points_offset = transformer.apply_cascade_offset(original_points, branch_hierarchy, offset_strength=0.4)
    
    # 2. 仅旋转变换
    print("2. 应用级联旋转...")
    points_rotation = transformer.apply_cascade_rotation(original_points, branch_hierarchy, rotation_strength=0.5)
    
    # 3. 组合变换
    print("3. 应用组合变换...")
    points_combined = transformer.apply_combined_transform(original_points, branch_hierarchy, 
                                                         offset_strength=0.3, rotation_strength=0.4)
    
    # 可视化结果
    print("\n📊 生成可视化...")
    
    # 偏移变换可视化
    visualize_cascade_transform(original_points, points_offset, branch_hierarchy, 
                               "级联偏移变换效果")
    
    # 旋转变换可视化
    visualize_cascade_transform(original_points, points_rotation, branch_hierarchy, 
                               "级联旋转变换效果")
    
    # 组合变换可视化
    visualize_cascade_transform(original_points, points_combined, branch_hierarchy, 
                               "组合变换效果（偏移+旋转）")
    
    print("\n✅ 演示完成！")
    return transformer, branch_hierarchy, original_points, points_combined


def apply_cascade_transform_to_branches(branches, max_offset=1.0, rotation_range=15.0, 
                                       offset_strength=0.3, rotation_strength=0.4):
    """
    对血管分支应用级联变换的实用函数
    
    Args:
        branches: 血管分支列表，每个分支包含points字段
        max_offset: 最大偏移距离
        rotation_range: 旋转角度范围（度）
        offset_strength: 偏移强度 (0-1)
        rotation_strength: 旋转强度 (0-1)
    
    Returns:
        transformed_branches: 变换后的分支列表
    """
    def transform_branch_points(points, is_main_branch=True):
        """对单个分支的点进行变换"""
        if len(points) < 2:
            return points
        
        points = np.array(points, dtype=np.float64)
        transformed_points = points.copy()
        if is_main_branch:
            # 主干使用较小的变换
            offset_factor = offset_strength * 0.5
            rotation_factor = rotation_strength * 0.5
        else:
            # 分支使用较大的变换
            offset_factor = offset_strength
            rotation_factor = rotation_strength
        
        # 级联偏移
        offset_direction = np.random.randn(3)
        offset_direction = offset_direction / np.linalg.norm(offset_direction)
        offset_magnitude = max_offset * offset_factor
        
        for i in range(len(points)):
            # 偏移量随距离衰减
            decay_factor = np.exp(-i * 0.1)
            current_offset = offset_direction * offset_magnitude * decay_factor
            transformed_points[i] += current_offset
            
            # 确保与前序节点的距离不超过限制
            if i > 0:
                distance = np.linalg.norm(transformed_points[i] - transformed_points[i-1])
                if distance > max_offset:
                    direction = (transformed_points[i] - transformed_points[i-1]) / distance
                    transformed_points[i] = transformed_points[i-1] + direction * max_offset
        
        # 级联旋转
        if len(points) >= 3:
            rotation_axis = np.random.randn(3)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.radians(rotation_range) * rotation_factor
            
            for i in range(1, len(points)):
                # 旋转角度随距离衰减
                decay_factor = np.exp(-i * 0.001)
                current_angle = rotation_angle * decay_factor
                
                # 获取前序节点作为旋转中心
                center = transformed_points[i-1]
                current_point = transformed_points[i]
                
                # 计算相对于中心的向量
                relative_vector = current_point - center
                
                # 应用旋转（使用罗德里格斯公式）
                cos_angle = np.cos(current_angle)
                sin_angle = np.sin(current_angle)
                
                rotated_vector = (relative_vector * cos_angle + 
                                np.cross(rotation_axis, relative_vector) * sin_angle + 
                                rotation_axis * np.dot(rotation_axis, relative_vector) * (1 - cos_angle))
                
                # 更新位置
                transformed_points[i] = center + rotated_vector
                
                # 确保距离限制
                distance = np.linalg.norm(rotated_vector)
                if distance > max_offset:
                    normalized_vector = rotated_vector / distance
                    transformed_points[i] = center + normalized_vector * max_offset
        
        return transformed_points.tolist()
    
    def transform_branch_recursive(branch):
        """递归变换分支及其子分支"""
        transformed_branch = branch.copy()
        
        # 变换当前分支的点
        if "points" in branch:
            is_main = "children" not in branch or len(branch.get("children", [])) > 0
            transformed_branch["points"] = transform_branch_points(branch["points"], is_main)
        
        # 递归变换子分支
        if "children" in branch:
            transformed_branch["children"] = [transform_branch_recursive(child) for child in branch["children"]]
        
        return transformed_branch
    
    # 对每个分支应用变换
    transformed_branches = [transform_branch_recursive(branch) for branch in branches]
    
    return transformed_branches


def quick_demo_cascade():
    """用forest_pointcloud.json中的血管树实例演示级联变换效果"""
    print("🚀 forest_pointcloud.json 血管树级联变换演示")
    import json
    # 读取forest_pointcloud.json
    with open('forest_pointcloud.json', 'r', encoding='utf-8') as f:
        forest = json.load(f)
    # 取第一个血管树实例
    tree = forest[0]
    branches = tree["branches"]
    print(f"读取到血管树: levels={tree.get('levels')}, branches数={len(branches)}")
    
    # 应用级联变换
    print("\n应用级联变换...")
    transformed_branches = apply_cascade_transform_to_branches(
        branches, 
        max_offset=20, 
        rotation_range=60.0,
        offset_strength=0.0,
        rotation_strength=0.5
    )
    
    # 提取所有点用于可视化
    def extract_all_points(branches):
        all_points = []
        def _extract(branch):
            if "points" in branch:
                all_points.extend(branch["points"])
            if "children" in branch:
                for child in branch["children"]:
                    _extract(child)
        for branch in branches:
            _extract(branch)
        return np.array(all_points)
    
    original_points = extract_all_points(branches)
    transformed_points = extract_all_points(transformed_branches)
    
    # 可视化对比
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 6))
    
    # 原始形状
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(*original_points.T, c='blue', s=10, alpha=0.8)
    ax1.set_title('原始血管形状')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 变换后形状
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(*transformed_points.T, c='red', s=10, alpha=0.8)
    ax2.set_title('变换后血管形状')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 对比视图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(*original_points.T, c='blue', s=8, alpha=0.5, label='原始')
    ax3.scatter(*transformed_points.T, c='red', s=8, alpha=0.5, label='变换后')
    for i in range(min(len(original_points), len(transformed_points))):
        ax3.plot([original_points[i, 0], transformed_points[i, 0]],
                [original_points[i, 1], transformed_points[i, 1]],
                [original_points[i, 2], transformed_points[i, 2]],
                'g-', alpha=0.1, linewidth=0.5)
    ax3.set_title('变换对比')
    ax3.legend()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # 统一视图范围
    all_points = np.vstack([original_points, transformed_points])
    center = all_points.mean(axis=0)
    range_val = max(np.std(all_points, axis=0)) * 3
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
    plt.suptitle('forest_pointcloud.json 血管树级联变换演示', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 计算统计信息
    offsets = np.linalg.norm(transformed_points - original_points, axis=1)
    print(f"\n📊 变换统计:")
    print(f"总点数: {len(original_points)}")
    print(f"平均偏移: {np.mean(offsets):.3f}")
    print(f"最大偏移: {np.max(offsets):.3f}")
    print(f"偏移标准差: {np.std(offsets):.3f}")
    
    return transformed_branches


# 如果直接运行此文件，执行快速演示
if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行快速演示
    quick_demo_cascade() 