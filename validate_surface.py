"""
血管曲面生成验证脚本
用于验证扩散模型生成的曲面质量
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tree_plane_diffusion import (
    CondNoisePredictor, 
    denoise_with_tree, 
    visualize_generated_surface,
    analyze_surface_quality,
    print_analysis_report,
    comprehensive_surface_validation,
    linear_beta_schedule
)
from tree_plane_predictor import tree_points_to_array
from visual import find_max_points_branches
import argparse
import os

def load_trained_model(model_path: str, grid_size: int = 32, device: str = 'cpu'):
    """加载训练好的模型"""
    model = CondNoisePredictor(grid_size=grid_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        betas = checkpoint.get('betas', linear_beta_schedule(100).to(device))
    else:
        # 如果只保存了模型参数
        model.load_state_dict(checkpoint)
        betas = linear_beta_schedule(100).to(device)
    
    model.eval()
    return model, betas

def compare_multiple_generations(tree_json: str, model, betas, n_samples: int = 5, grid_size: int = 32, device: str = 'cpu'):
    """
    生成多个样本并比较差异
    """
    print(f"生成 {n_samples} 个曲面样本进行比较...")
    
    samples = []
    for i in range(n_samples):
        print(f"生成样本 {i+1}/{n_samples}...")
        sample = denoise_with_tree(tree_json, model, betas, device=device, grid_size=grid_size)
        samples.append(sample)
    
    # 可视化多个样本
    fig = plt.figure(figsize=(20, 4))
    
    # 读取原始数据用于参考
    with open(tree_json, 'r') as fp:
        td = json.load(fp)
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(td)
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    center = all_points.mean(axis=0)
    range_val = 30
    
    for i, sample in enumerate(samples):
        ax = fig.add_subplot(1, n_samples, i+1, projection='3d')
        
        # 绘制原始血管点
        ax.scatter(*trunk_pts.T, c='blue', s=1, alpha=0.3)
        ax.scatter(*br1_pts.T, c='green', s=1, alpha=0.3)
        ax.scatter(*br2_pts.T, c='red', s=1, alpha=0.3)
        
        # 绘制生成的曲面
        X, Y, Z = sample[:, :, 0], sample[:, :, 1], sample[:, :, 2]
        ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
        
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
        ax.set_title(f'样本 {i+1}')
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig('multiple_samples_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算样本间的差异
    print("\n样本间差异分析:")
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            diff = np.mean(np.linalg.norm(samples[i] - samples[j], axis=-1))
            print(f"样本 {i+1} vs 样本 {j+1}: 平均点距差异 = {diff:.4f}")
    
    return samples

def detailed_surface_analysis(tree_json: str, surface_points: np.ndarray):
    """
    详细的曲面分析，包括几何特性
    """
    print("\n详细曲面几何分析:")
    print("-" * 40)
    
    grid_size = surface_points.shape[0]
    
    # 1. 曲率分析
    curvatures = []
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            # 计算局部曲率（二阶导数的近似）
            p_center = surface_points[i, j]
            p_left = surface_points[i, j-1]
            p_right = surface_points[i, j+1]
            p_up = surface_points[i-1, j]
            p_down = surface_points[i+1, j]
            
            # 二阶差分近似曲率
            curvature_x = np.linalg.norm(p_left - 2*p_center + p_right)
            curvature_y = np.linalg.norm(p_up - 2*p_center + p_down)
            total_curvature = (curvature_x + curvature_y) / 2
            curvatures.append(total_curvature)
    
    print(f"1. 曲率分析:")
    print(f"   平均曲率: {np.mean(curvatures):.6f}")
    print(f"   曲率标准差: {np.std(curvatures):.6f}")
    print(f"   最大曲率: {np.max(curvatures):.6f}")
    
    # 2. 表面积估算
    total_area = 0
    for i in range(grid_size-1):
        for j in range(grid_size-1):
            # 每个小方格分为两个三角形计算面积
            p1 = surface_points[i, j]
            p2 = surface_points[i+1, j]
            p3 = surface_points[i, j+1]
            p4 = surface_points[i+1, j+1]
            
            # 三角形1的面积
            area1 = 0.5 * np.linalg.norm(np.cross(p2-p1, p3-p1))
            # 三角形2的面积
            area2 = 0.5 * np.linalg.norm(np.cross(p4-p2, p4-p3))
            
            total_area += area1 + area2
    
    print(f"\n2. 表面积估算:")
    print(f"   总表面积: {total_area:.4f}")
    
    # 3. 厚度分析（如果是弯曲表面）
    center_line = surface_points[grid_size//2, :]  # 中间一行
    thicknesses = []
    for i in range(1, len(center_line)-1):
        # 计算局部厚度（相邻点到中心点的距离）
        thickness = np.linalg.norm(center_line[i-1] - center_line[i+1])
        thicknesses.append(thickness)
    
    print(f"\n3. 局部厚度分析:")
    print(f"   平均厚度: {np.mean(thicknesses):.4f}")
    print(f"   厚度变化: {np.std(thicknesses):.4f}")

def validate_with_different_trees(model_path: str, tree_files: list, grid_size: int = 32, device: str = 'cpu'):
    """
    用不同的血管树测试模型的泛化能力
    """
    print(f"使用 {len(tree_files)} 个不同血管树测试模型泛化能力...")
    
    model, betas = load_trained_model(model_path, grid_size, device)
    
    results = {}
    
    for i, tree_file in enumerate(tree_files):
        print(f"\n测试血管树 {i+1}: {tree_file}")
        print("-" * 50)
        
        try:
            # 生成曲面
            surface = denoise_with_tree(tree_file, model, betas, device=device, grid_size=grid_size)
            
            # 分析质量
            metrics = analyze_surface_quality(tree_file, surface)
            results[tree_file] = metrics
            
            # 简要报告
            print(f"中心距离: {metrics['center_distance']:.4f}")
            print(f"平均拟合距离: {metrics['min_distance_stats']['mean']:.4f}")
            print(f"覆盖比例: {metrics['coverage_ratio']}")
            
            # 保存可视化
            base_name = os.path.splitext(os.path.basename(tree_file))[0]
            visualize_generated_surface(tree_file, surface, f"validation_{base_name}.png")
            
        except Exception as e:
            print(f"处理 {tree_file} 时出错: {e}")
            results[tree_file] = None
    
    # 汇总分析
    print("\n" + "="*60)
    print("泛化能力汇总报告")
    print("="*60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # 计算各指标的统计
        center_distances = [r['center_distance'] for r in valid_results.values()]
        fit_distances = [r['min_distance_stats']['mean'] for r in valid_results.values()]
        
        print(f"测试样本数: {len(valid_results)}")
        print(f"中心距离 - 平均: {np.mean(center_distances):.4f}, 标准差: {np.std(center_distances):.4f}")
        print(f"拟合距离 - 平均: {np.mean(fit_distances):.4f}, 标准差: {np.std(fit_distances):.4f}")
    
    return results

def generate_ideal_surface_from_tree(tree_json: str, grid_size: int = 32):
    """
    基于血管树的分支中点和主干点生成理想曲面
    
    Args:
        tree_json: 血管树JSON文件路径
        grid_size: 网格大小
    
    Returns:
        ideal_surface: 理想曲面点，形状为(grid_size, grid_size, 3)
    """
    import json
    
    # 读取血管数据
    with open(tree_json, 'r') as fp:
        td = json.load(fp)
    
    # 修复：正确处理数据结构
    def safe_find_max_points_branches(tree_data):
        """安全的分支点提取函数，处理不同的数据结构"""
        try:
            # 尝试使用原始函数
            from visual import find_max_points_branches
            return find_max_points_branches(tree_data)
        except (KeyError, TypeError, IndexError) as e:
            print(f"原始函数失败: {e}")
            print("使用备用方法提取分支点...")
            
            # 备用方法：直接从branches列表中提取
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
            print("所有方法都失败，创建默认分支点...")
            default_points = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 2]], dtype=np.float32)
            return default_points, default_points, default_points
    
    # 获取主干和分支点
    trunk_pts, br1_pts, br2_pts = safe_find_max_points_branches(td)
    
    print(f"提取到的点数: 主干={len(trunk_pts)}, 分支1={len(br1_pts)}, 分支2={len(br2_pts)}")
    
    # 计算两个分支对应点的中点
    min_len = min(len(br1_pts), len(br2_pts))
    if min_len > 0:
        # 取相同数量的点来计算中点
        br1_sampled = br1_pts[:min_len]
        br2_sampled = br2_pts[:min_len]
        midpoints = (br1_sampled + br2_sampled) / 2.0
    else:
        # 如果没有分支点，使用主干末端作为中点
        midpoints = np.array([trunk_pts[-1]])
    
    # 合并主干点和中点来拟合曲面
    surface_points = np.vstack([trunk_pts, midpoints])
    
    # 使用PCA找到主要方向
    centroid = surface_points.mean(axis=0)
    centered_points = surface_points - centroid
    
    # 计算协方差矩阵和主成分
    cov_matrix = np.cov(centered_points.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # 按特征值排序（降序）
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # 主方向和次方向
    primary_dir = eigenvecs[:, 0]    # 最大特征值对应的方向
    secondary_dir = eigenvecs[:, 1]  # 第二大特征值对应的方向
    
    # 计算曲面的范围
    proj_primary = np.dot(centered_points, primary_dir)
    proj_secondary = np.dot(centered_points, secondary_dir)
    
    primary_range = [proj_primary.min(), proj_primary.max()]
    secondary_range = [proj_secondary.min(), proj_secondary.max()]
    
    # 在主平面上生成32x32网格点
    u = np.linspace(primary_range[0], primary_range[1], grid_size)
    v = np.linspace(secondary_range[0], secondary_range[1], grid_size)
    U, V = np.meshgrid(u, v)
    
    # 将网格点转换回3D空间
    ideal_surface = np.zeros((grid_size, grid_size, 3))
    for i in range(grid_size):
        for j in range(grid_size):
            # 在主平面上的点
            point_2d = U[i,j] * primary_dir + V[i,j] * secondary_dir
            
            # 添加第三维的弯曲（基于到中心的距离）
            dist_from_center = np.sqrt(U[i,j]**2 + V[i,j]**2)
            curvature = 0.1 * dist_from_center * np.sin(dist_from_center * 0.1)  # 轻微弯曲
            
            # 第三个方向（法向量）
            normal_dir = eigenvecs[:, 2]
            point_3d = centroid + point_2d + curvature * normal_dir
            
            ideal_surface[i, j] = point_3d
    
    return ideal_surface

def compare_ideal_vs_generated(tree_json: str, generated_surface: np.ndarray, save_prefix: str = "comparison"):
    """
    对比理想曲面与生成曲面
    
    Args:
        tree_json: 血管树JSON文件路径
        generated_surface: 生成的曲面，形状为(grid_size, grid_size, 3)
        save_prefix: 保存文件前缀
    """
    import matplotlib.pyplot as plt
    
    # 生成理想曲面
    grid_size = generated_surface.shape[0]
    ideal_surface = generate_ideal_surface_from_tree(tree_json, grid_size)
    
    # 读取血管数据用于可视化
    with open(tree_json, 'r') as fp:
        td = json.load(fp)
    
    # 使用安全的分支点提取方法
    def safe_find_max_points_branches(tree_data):
        """安全的分支点提取函数，处理不同的数据结构"""
        try:
            # 尝试使用原始函数
            from visual import find_max_points_branches
            return find_max_points_branches(tree_data)
        except (KeyError, TypeError, IndexError) as e:
            print(f"原始函数失败: {e}")
            print("使用备用方法提取分支点...")
            
            # 备用方法：直接从branches列表中提取
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
            print("所有方法都失败，创建默认分支点...")
            default_points = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 2]], dtype=np.float32)
            return default_points, default_points, default_points
    
    trunk_pts, br1_pts, br2_pts = safe_find_max_points_branches(td)
    
    # 创建对比图
    fig = plt.figure(figsize=(20, 5))
    
    # 统一视图范围
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    center = all_points.mean(axis=0)
    range_val = 30
    
    # 子图1: 理想曲面
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(*trunk_pts.T, c='blue', s=2, alpha=0.6, label='主干')
    ax1.scatter(*br1_pts.T, c='green', s=2, alpha=0.6, label='分支1')
    ax1.scatter(*br2_pts.T, c='red', s=2, alpha=0.6, label='分支2')
    
    X_ideal, Y_ideal, Z_ideal = ideal_surface[:, :, 0], ideal_surface[:, :, 1], ideal_surface[:, :, 2]
    ax1.plot_surface(X_ideal, Y_ideal, Z_ideal, alpha=0.7, cmap='coolwarm')
    ax1.set_title('理想曲面\n(基于分支中点+主干)')
    ax1.set_xlim(center[0] - range_val, center[0] + range_val)
    ax1.set_ylim(center[1] - range_val, center[1] + range_val)
    ax1.set_zlim(center[2] - range_val, center[2] + range_val)
    ax1.set_axis_off()
    
    # 子图2: 生成曲面
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(*trunk_pts.T, c='blue', s=2, alpha=0.6, label='主干')
    ax2.scatter(*br1_pts.T, c='green', s=2, alpha=0.6, label='分支1')
    ax2.scatter(*br2_pts.T, c='red', s=2, alpha=0.6, label='分支2')
    
    X_gen, Y_gen, Z_gen = generated_surface[:, :, 0], generated_surface[:, :, 1], generated_surface[:, :, 2]
    ax2.plot_surface(X_gen, Y_gen, Z_gen, alpha=0.7, cmap='viridis')
    ax2.set_title('扩散模型生成曲面')
    ax2.set_xlim(center[0] - range_val, center[0] + range_val)
    ax2.set_ylim(center[1] - range_val, center[1] + range_val)
    ax2.set_zlim(center[2] - range_val, center[2] + range_val)
    ax2.set_axis_off()
    
    # 子图3: 差异可视化
    ax3 = fig.add_subplot(143, projection='3d')
    difference = np.linalg.norm(ideal_surface - generated_surface, axis=-1)
    
    # 绘制血管点
    ax3.scatter(*trunk_pts.T, c='blue', s=1, alpha=0.3)
    ax3.scatter(*br1_pts.T, c='green', s=1, alpha=0.3)
    ax3.scatter(*br2_pts.T, c='red', s=1, alpha=0.3)
    
    # 绘制差异热图
    U_grid = np.arange(grid_size)
    V_grid = np.arange(grid_size)
    U_mesh, V_mesh = np.meshgrid(U_grid, V_grid)
    
    # 使用差异值作为颜色映射
    surf = ax3.scatter(X_gen.flatten(), Y_gen.flatten(), Z_gen.flatten(), 
                      c=difference.flatten(), cmap='hot', s=5, alpha=0.8)
    
    ax3.set_title('点对点差异\n(颜色越热差异越大)')
    ax3.set_xlim(center[0] - range_val, center[0] + range_val)
    ax3.set_ylim(center[1] - range_val, center[1] + range_val)
    ax3.set_zlim(center[2] - range_val, center[2] + range_val)
    ax3.set_axis_off()
    plt.colorbar(surf, ax=ax3, shrink=0.8)
    
    # 子图4: 叠加对比
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(*trunk_pts.T, c='blue', s=1, alpha=0.3)
    ax4.scatter(*br1_pts.T, c='green', s=1, alpha=0.3)
    ax4.scatter(*br2_pts.T, c='red', s=1, alpha=0.3)
    
    # 理想曲面（半透明蓝色）
    ax4.plot_surface(X_ideal, Y_ideal, Z_ideal, alpha=0.3, color='blue', label='理想')
    # 生成曲面（半透明橙色）
    ax4.plot_surface(X_gen, Y_gen, Z_gen, alpha=0.3, color='orange', label='生成')
    
    ax4.set_title('叠加对比\n(蓝色:理想 橙色:生成)')
    ax4.set_xlim(center[0] - range_val, center[0] + range_val)
    ax4.set_ylim(center[1] - range_val, center[1] + range_val)
    ax4.set_zlim(center[2] - range_val, center[2] + range_val)
    ax4.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_ideal_vs_generated.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算数值差异指标
    diff_metrics = {
        'mean_point_distance': np.mean(difference),
        'std_point_distance': np.std(difference),
        'max_point_distance': np.max(difference),
        'median_point_distance': np.median(difference),
        'rmse': np.sqrt(np.mean(difference**2))
    }
    
    print(f"\n理想曲面 vs 生成曲面 差异分析:")
    print(f"平均点距差异: {diff_metrics['mean_point_distance']:.4f}")
    print(f"差异标准差: {diff_metrics['std_point_distance']:.4f}")
    print(f"最大点距差异: {diff_metrics['max_point_distance']:.4f}")
    print(f"差异中位数: {diff_metrics['median_point_distance']:.4f}")
    print(f"均方根误差: {diff_metrics['rmse']:.4f}")
    
    # 保存差异指标
    import json
    with open(f"{save_prefix}_ideal_comparison_metrics.json", 'w') as f:
        json.dump(diff_metrics, f, indent=2)
    
    print(f"对比结果已保存:")
    print(f"- {save_prefix}_ideal_vs_generated.png")
    print(f"- {save_prefix}_ideal_comparison_metrics.json")
    
    return ideal_surface, diff_metrics

def validate_ideal_surface_only(tree_json: str, grid_size: int = 32, save_prefix: str = "ideal_surface"):
    """
    只验证理想曲面（不需要训练模型）
    
    Args:
        tree_json: 血管树JSON文件路径
        grid_size: 网格大小
        save_prefix: 保存文件前缀
    """
    print(f"生成基于分支中点和主干的理想曲面...")
    
    # 生成理想曲面
    ideal_surface = generate_ideal_surface_from_tree(tree_json, grid_size)
    
    # 可视化理想曲面
    print("可视化理想曲面...")
    visualize_generated_surface(
        tree_json, 
        ideal_surface, 
        save_path=f"{save_prefix}_visualization.png",
        show_wireframe=True
    )
    
    # 分析理想曲面质量
    print("分析理想曲面质量...")
    metrics = analyze_surface_quality(tree_json, ideal_surface)
    print_analysis_report(metrics)
    
    # 保存详细分析
    comprehensive_surface_validation(tree_json, ideal_surface, save_prefix)
    
    print(f"\n理想曲面验证完成！")
    print(f"这个曲面是基于以下数据生成的:")
    print(f"- 血管主干点")
    print(f"- 两个分支对应点的中点")
    print(f"- PCA主成分分析确定的主平面")
    print(f"- 轻微的距离相关弯曲")
    
    return ideal_surface

def main():
    parser = argparse.ArgumentParser(description='血管曲面生成验证')
    parser.add_argument('--tree_json', type=str, required=True, help='血管树JSON文件路径')
    parser.add_argument('--model_path', type=str, help='训练好的模型路径（可选）')
    parser.add_argument('--grid_size', type=int, default=32, help='网格大小')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--mode', type=str, default='single', 
                      choices=['single', 'multiple', 'detailed', 'generalization', 'ideal_surface', 'ideal_comparison'],
                      help='验证模式')
    parser.add_argument('--n_samples', type=int, default=5, help='多样本模式下的样本数')
    parser.add_argument('--output_prefix', type=str, default='validation', help='输出文件前缀')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # 单个曲面验证
        if args.model_path:
            model, betas = load_trained_model(args.model_path, args.grid_size, args.device)
            surface = denoise_with_tree(args.tree_json, model, betas, 
                                      device=args.device, grid_size=args.grid_size)
        else:
            print("单个验证模式需要提供模型路径")
            return
        
        comprehensive_surface_validation(args.tree_json, surface, args.output_prefix)
        
    elif args.mode == 'multiple':
        # 多样本比较
        if not args.model_path:
            print("多样本模式需要提供模型路径")
            return
        
        model, betas = load_trained_model(args.model_path, args.grid_size, args.device)
        samples = compare_multiple_generations(args.tree_json, model, betas, 
                                             args.n_samples, args.grid_size, args.device)
        
        # 对第一个样本做详细分析
        comprehensive_surface_validation(args.tree_json, samples[0], f"{args.output_prefix}_sample1")
        
    elif args.mode == 'detailed':
        # 详细几何分析
        if not args.model_path:
            print("详细分析模式需要提供模型路径")
            return
        
        model, betas = load_trained_model(args.model_path, args.grid_size, args.device)
        surface = denoise_with_tree(args.tree_json, model, betas, 
                                  device=args.device, grid_size=args.grid_size)
        
        comprehensive_surface_validation(args.tree_json, surface, args.output_prefix)
        detailed_surface_analysis(args.tree_json, surface)
        
    elif args.mode == 'ideal_surface':
        # 理想曲面验证（不需要模型）
        ideal_surface = validate_ideal_surface_only(args.tree_json, args.grid_size, args.output_prefix)
        
    elif args.mode == 'ideal_comparison':
        # 理想曲面与生成曲面对比
        if not args.model_path:
            print("理想对比模式需要提供模型路径")
            return
        
        model, betas = load_trained_model(args.model_path, args.grid_size, args.device)
        generated_surface = denoise_with_tree(args.tree_json, model, betas, 
                                            device=args.device, grid_size=args.grid_size)
        
        # 进行对比分析
        ideal_surface, diff_metrics = compare_ideal_vs_generated(
            args.tree_json, generated_surface, args.output_prefix
        )
        
    elif args.mode == 'generalization':
        # 泛化能力测试
        if not args.model_path:
            print("泛化测试模式需要提供模型路径")
            return
        
        import glob
        tree_files = glob.glob('tree_*.json')
        if len(tree_files) < 2:
            print("泛化测试需要至少2个血管树文件")
            return
        
        results = validate_with_different_trees(args.model_path, tree_files, 
                                              args.grid_size, args.device)

if __name__ == '__main__':
    main() 