#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface Points Visualization Tool
可视化TempDataset类生成的surface_points

功能：
- 显示原始血管树（主干和分支）
- 显示生成的曲面网格点
- 支持交互式3D查看（鼠标旋转、缩放）
- 多角度对比展示
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob


# ========== 数据处理函数 ==========

def tree_points_to_array(tree_data):
    """将血管树数据转换为numpy数组"""
    points = []
    
    def extract_points_recursive(branch, branch_id=0, parent_id=-1):
        if "points" in branch:
            branch_points = np.array(branch["points"], dtype=np.float32)
            # 添加分支ID和父分支ID
            ids = np.full((len(branch_points), 2), [branch_id, parent_id], dtype=np.float32)
            branch_with_ids = np.hstack([branch_points, ids])
            points.append(branch_with_ids)
        
        if "children" in branch:
            for i, child in enumerate(branch["children"]):
                extract_points_recursive(child, branch_id + i + 1, branch_id)
    
    if "branches" in tree_data and len(tree_data["branches"]) > 0:
        extract_points_recursive(tree_data["branches"][0])
    
    if points:
        return np.vstack(points)
    else:
        return np.array([]).reshape(0, 5)


def safe_find_max_points_branches(tree_data):
    """安全的分支点提取函数"""
    try:
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
        print("创建默认分支点数据...")
        default_points = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 2]], dtype=np.float32)
        return default_points, default_points, default_points
        
    except Exception as e:
        print(f"提取分支点失败: {e}")
        default_points = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 2]], dtype=np.float32)
        return default_points, default_points, default_points


# ========== 曲面生成类 ==========

class TempDataset:
    """临时数据集类，用于生成曲面网格点"""
    
    def _generate_surface_grid(self, centerline_points, main_direction, grid_size, point_spacing):
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
            axis_pos = self._interpolate_on_centerline(centerline_points, t)
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
    
    def _interpolate_on_centerline(self, centerline_points, t):
        """在中轴线上按参数t插值 (t在0到1之间)"""
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


# ========== 可视化函数 ==========

def visualize_surface_with_vessels(tree_json, grid_size=32, point_spacing=0.2):
    """
    可视化TempDataset生成的曲面与原始血管
    
    Args:
        tree_json: 血管树json文件路径
        grid_size: 网格大小
        point_spacing: 点间距
    """
    print(f"正在处理血管树文件: {tree_json}")
    
    # 读取血管树数据
    with open(tree_json, 'r') as fp:
        td = json.load(fp)
    
    # 提取血管点
    trunk_pts, br1_pts, br2_pts = safe_find_max_points_branches(td)
    print(f"提取到血管点 - 主干: {len(trunk_pts)}, 分支1: {len(br1_pts)}, 分支2: {len(br2_pts)}")
    
    # 构建中轴线
    min_len = min(len(br1_pts), len(br2_pts))
    if min_len > 0:
        br1_sampled = br1_pts[:min_len]
        br2_sampled = br2_pts[:min_len]
        midpoints = (br1_sampled + br2_sampled) / 2.0
    else:
        midpoints = np.array([trunk_pts[-1]])
    
    # 构建中轴线：主干点 + 分支中点
    centerline_points = np.vstack([trunk_pts, midpoints])
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
    
    print(f"中轴线信息: {len(sorted_centerline)} 个点，主方向: {main_direction}")
    
    # 生成曲面网格点
    temp_dataset = TempDataset()
    surface_points = temp_dataset._generate_surface_grid(
        sorted_centerline, main_direction, grid_size, point_spacing
    )
    
    print(f"生成曲面网格: {surface_points.shape}")
    
    # 创建交互式可视化
    fig = plt.figure(figsize=(20, 10))
    
    # 子图1: 总体视图
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 绘制血管点
    ax1.scatter(*trunk_pts.T, c='blue', s=4, alpha=0.8, label='主干', marker='o')
    ax1.scatter(*br1_pts.T, c='green', s=6, alpha=0.9, label='分支1', marker='^')
    ax1.scatter(*br2_pts.T, c='red', s=6, alpha=0.9, label='分支2', marker='s')
    ax1.plot(*sorted_centerline.T, 'purple', linewidth=3, alpha=0.8, label='中轴线')
    
    # 绘制曲面
    X, Y, Z = surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2]
    ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis', linewidth=0.1)
    
    ax1.set_title(f'总体视图\n网格: {grid_size}×{grid_size}, 间距: {point_spacing}')
    ax1.legend()
    
    # 子图2: 仅显示曲面
    ax2 = fig.add_subplot(222, projection='3d')
    
    surf = ax2.plot_surface(X, Y, Z, alpha=0.8, cmap='plasma', linewidth=0.1)
    ax2.plot(*sorted_centerline.T, 'black', linewidth=4, alpha=1.0, label='中轴线')
    
    # 绘制网格线
    for i in range(0, grid_size, 4):
        ax2.plot(X[i, :], Y[i, :], Z[i, :], 'k-', alpha=0.5, linewidth=1)
    for j in range(0, grid_size, 4):
        ax2.plot(X[:, j], Y[:, j], Z[:, j], 'k-', alpha=0.5, linewidth=1)
    
    ax2.set_title('曲面详细视图\n(显示网格结构)')
    ax2.legend()
    
    # 子图3: 网格点分布
    ax3 = fig.add_subplot(223, projection='3d')
    
    # 显示网格点
    surface_flat = surface_points.reshape(-1, 3)
    ax3.scatter(*surface_flat.T, c='orange', s=3, alpha=0.8, label='曲面网格点')
    ax3.scatter(*trunk_pts.T, c='blue', s=2, alpha=0.6, label='主干')
    ax3.scatter(*br1_pts.T, c='green', s=3, alpha=0.7, label='分支1')
    ax3.scatter(*br2_pts.T, c='red', s=3, alpha=0.7, label='分支2')
    ax3.plot(*sorted_centerline.T, 'purple', linewidth=2, alpha=0.8, label='中轴线')
    
    ax3.set_title('网格点分布')
    ax3.legend()
    
    # 子图4: 切片视图
    ax4 = fig.add_subplot(224, projection='3d')
    
    # 显示几个代表性的切片
    slice_indices = [0, grid_size//4, grid_size//2, 3*grid_size//4, grid_size-1]
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    
    for i, color in zip(slice_indices, colors):
        slice_points = surface_points[i, :, :]
        ax4.plot(slice_points[:, 0], slice_points[:, 1], slice_points[:, 2], 
                color=color, linewidth=3, alpha=0.8, label=f'切片 {i}')
    
    ax4.scatter(*trunk_pts.T, c='lightblue', s=2, alpha=0.4)
    ax4.plot(*sorted_centerline.T, 'black', linewidth=3, alpha=0.8, label='中轴线')
    
    ax4.set_title('曲面切片视图')
    ax4.legend()
    
    # 统一设置视图范围
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts, surface_flat])
    center = all_points.mean(axis=0)
    range_val = max(np.std(all_points, axis=0)) * 3
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    
    # 打印统计信息
    print("\n" + "="*50)
    print("曲面生成统计信息")
    print("="*50)
    print(f"血管树文件: {tree_json}")
    print(f"主干点数: {len(trunk_pts)}")
    print(f"分支1点数: {len(br1_pts)}")
    print(f"分支2点数: {len(br2_pts)}")
    print(f"中轴线点数: {len(sorted_centerline)}")
    print(f"生成的曲面网格: {grid_size} × {grid_size}")
    print(f"总网格点数: {grid_size * grid_size}")
    print(f"点间距: {point_spacing}")
    print(f"曲面范围: {(grid_size-1)*point_spacing:.2f}")
    print(f"曲面中心: {surface_points.mean(axis=(0,1))}")
    print("="*50)
    
    # 显示交互式界面
    print("\n🎮 交互式3D可视化已启动!")
    print("操作说明:")
    print("- 鼠标左键拖拽: 旋转视角")
    print("- 鼠标滚轮: 缩放")
    print("- 鼠标右键拖拽: 平移")
    print("- 关闭窗口: 结束查看")
    
    plt.show()
    
    return surface_points, sorted_centerline


def create_single_view_visualization(tree_json, grid_size=32, point_spacing=0.2):
    """创建单一大视图的可视化"""
    
    print(f"创建单一视图可视化: {tree_json}")
    
    # 读取和处理数据（代码与上面相同）
    with open(tree_json, 'r') as fp:
        td = json.load(fp)
    
    trunk_pts, br1_pts, br2_pts = safe_find_max_points_branches(td)
    
    # 构建中轴线
    min_len = min(len(br1_pts), len(br2_pts))
    if min_len > 0:
        br1_sampled = br1_pts[:min_len]
        br2_sampled = br2_pts[:min_len]
        midpoints = (br1_sampled + br2_sampled) / 2.0
    else:
        midpoints = np.array([trunk_pts[-1]])
    
    centerline_points = np.vstack([trunk_pts, midpoints])
    centerline_center = centerline_points.mean(axis=0)
    centerline_centered = centerline_points - centerline_center
    
    cov_matrix = np.cov(centerline_centered.T)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    idx = np.argsort(eigenvals)[::-1]
    main_direction = eigenvecs[:, idx[0]]
    
    projections = np.dot(centerline_centered, main_direction)
    sorted_indices = np.argsort(projections)
    sorted_centerline = centerline_points[sorted_indices]
    
    # 生成曲面
    temp_dataset = TempDataset()
    surface_points = temp_dataset._generate_surface_grid(
        sorted_centerline, main_direction, grid_size, point_spacing
    )
    
    # 创建单一大视图
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制血管点
    ax.scatter(*trunk_pts.T, c='navy', s=8, alpha=0.8, label='主干', marker='o')
    ax.scatter(*br1_pts.T, c='forestgreen', s=12, alpha=0.9, label='分支1', marker='^')
    ax.scatter(*br2_pts.T, c='crimson', s=12, alpha=0.9, label='分支2', marker='s')
    
    # 绘制中轴线
    ax.plot(*sorted_centerline.T, 'purple', linewidth=4, alpha=0.9, label='中轴线')
    
    # 绘制曲面
    X, Y, Z = surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2]
    surf = ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', linewidth=0.1)
    
    # 绘制网格线
    for i in range(0, grid_size, 2):
        ax.plot(X[i, :], Y[i, :], Z[i, :], 'yellow', alpha=0.6, linewidth=1)
    for j in range(0, grid_size, 2):
        ax.plot(X[:, j], Y[:, j], Z[:, j], 'yellow', alpha=0.6, linewidth=1)
    
    # 设置标题和标签
    ax.set_title(f'TempDataset生成的曲面可视化\n网格: {grid_size}×{grid_size}, 间距: {point_spacing}', 
                fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.set_zlabel('Z坐标', fontsize=12)
    
    # 设置视图范围
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts, surface_points.reshape(-1, 3)])
    center = all_points.mean(axis=0)
    range_val = max(np.std(all_points, axis=0)) * 3
    
    ax.set_xlim(center[0] - range_val, center[0] + range_val)
    ax.set_ylim(center[1] - range_val, center[1] + range_val)
    ax.set_zlim(center[2] - range_val, center[2] + range_val)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()
    
    return surface_points


# ========== 主函数 ==========

def main():
    """主函数"""
    print("🎯 TempDataset Surface Points 可视化工具")
    print("="*60)
    
    # 查找血管树文件
    json_files = glob.glob('tree_*.json')
    
    if not json_files:
        print("❌ 未找到血管树文件 (tree_*.json)")
        print("请确保当前目录下有血管树JSON文件")
        return
    
    print(f"📁 找到 {len(json_files)} 个血管树文件:")
    for i, file in enumerate(json_files):
        print(f"  {i+1}. {file}")
    
    # 选择文件
    if len(json_files) == 1:
        selected_file = json_files[0]
        print(f"\n🎯 自动选择文件: {selected_file}")
    else:
        try:
            choice = int(input(f"\n请选择文件编号 (1-{len(json_files)}): ")) - 1
            if 0 <= choice < len(json_files):
                selected_file = json_files[choice]
            else:
                print("❌ 无效选择，使用第一个文件")
                selected_file = json_files[0]
        except ValueError:
            print("❌ 输入无效，使用第一个文件")
            selected_file = json_files[0]
    
    # 参数设置
    print(f"\n⚙️ 参数设置:")
    
    try:
        grid_size = int(input("网格大小 (默认32): ") or "32")
    except ValueError:
        grid_size = 32
    
    try:
        point_spacing = float(input("点间距 (默认0.2): ") or "0.2")
    except ValueError:
        point_spacing = 0.2
    
    # 可视化模式选择
    print(f"\n🎨 可视化模式:")
    print("1. 四视图模式 (推荐)")
    print("2. 单一大视图模式")
    
    try:
        vis_mode = int(input("请选择模式 (1或2，默认1): ") or "1")
    except ValueError:
        vis_mode = 1
    
    print(f"\n🚀 开始生成可视化...")
    print(f"文件: {selected_file}")
    print(f"网格大小: {grid_size}×{grid_size}")
    print(f"点间距: {point_spacing}")
    
    # 执行可视化
    try:
        if vis_mode == 2:
            surface_points = create_single_view_visualization(selected_file, grid_size, point_spacing)
        else:
            surface_points, centerline = visualize_surface_with_vessels(selected_file, grid_size, point_spacing)
        
        print(f"\n✅ 可视化完成!")
        print(f"生成了 {surface_points.shape[0]}×{surface_points.shape[1]} 的曲面网格")
        
    except Exception as e:
        print(f"\n❌ 可视化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main() 