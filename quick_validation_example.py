"""
快速验证示例脚本
演示如何验证扩散模型生成的血管曲面
"""

import numpy as np
import glob
from tree_plane_diffusion import (
    train_tree_diffusion, 
    denoise_with_tree, 
    comprehensive_surface_validation,
    visualize_generated_surface,
    analyze_surface_quality,
    print_analysis_report
)
from validate_surface import (
    generate_ideal_surface_from_tree,
    validate_ideal_surface_only,
    compare_ideal_vs_generated
)

def quick_validation_demo():
    """快速验证演示"""
    print("血管曲面生成验证演示")
    print("=" * 50)
    
    # 1. 查找血管树文件
    tree_files = glob.glob('tree_*.json')
    if not tree_files:
        print("错误：未找到血管树文件 (tree_*.json)")
        print("请确保当前目录下有血管树JSON文件")
        return
    
    print(f"找到 {len(tree_files)} 个血管树文件: {tree_files}")
    
    # 2. 训练简单模型（用于演示）
    print("\n步骤1: 训练演示模型（轻量级）...")
    grid_size = 16  # 使用较小的网格加快演示
    model, betas = train_tree_diffusion(
        tree_files, 
        epochs=1000,  # 少量epoch用于演示
        batch_size=1, 
        device='cpu', 
        grid_size=grid_size
    )
    
    # 3. 生成曲面
    print("\n步骤2: 生成曲面...")
    test_file = tree_files[0]
    generated_surface = denoise_with_tree(
        test_file, 
        model, 
        betas, 
        device='cpu', 
        grid_size=grid_size
    )
    
    print(f"生成的曲面形状: {generated_surface.shape}")
    
    # 4. 基础可视化
    print("\n步骤3: 基础可视化...")
    visualize_generated_surface(
        test_file, 
        generated_surface, 
        save_path="demo_surface_visualization.png"
    )
    
    # 5. 质量分析
    print("\n步骤4: 质量分析...")
    metrics = analyze_surface_quality(test_file, generated_surface)
    print_analysis_report(metrics)
    
    # 6. 综合验证
    print("\n步骤5: 综合验证...")
    comprehensive_surface_validation(
        test_file, 
        generated_surface, 
        save_prefix="demo_validation"
    )
    
    print("\n验证完成！生成的文件:")
    print("- demo_surface_visualization.png: 基础可视化")
    print("- demo_validation_visualization.png: 综合可视化")
    print("- demo_validation_metrics.json: 详细质量指标")

def interactive_validation():
    """交互式验证"""
    tree_files = glob.glob('tree_*.json')
    if not tree_files:
        print("未找到血管树文件")
        return
    
    print("交互式曲面验证")
    print("可用的血管树文件:")
    for i, f in enumerate(tree_files):
        print(f"{i+1}. {f}")
    
    try:
        choice = int(input("选择要验证的文件编号: ")) - 1
        if 0 <= choice < len(tree_files):
            selected_file = tree_files[choice]
            
            # 生成随机曲面进行演示（实际使用中应该是训练好的模型）
            print(f"正在为 {selected_file} 生成演示曲面...")
            
            # 创建一个简单的演示曲面
            grid_size = 20
            demo_surface = create_demo_surface(selected_file, grid_size)
            
            print("进行可视化和分析...")
            comprehensive_surface_validation(
                selected_file, 
                demo_surface, 
                save_prefix=f"interactive_{choice+1}"
            )
            
        else:
            print("无效的选择")
    except ValueError:
        print("请输入有效的数字")

def create_demo_surface(tree_json: str, grid_size: int = 20):
    """
    创建一个演示用的曲面（用于测试验证功能）
    """
    import json
    
    # 读取血管数据
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
    all_points = np.vstack([trunk_pts, br1_pts, br2_pts])
    
    # 计算血管的主要方向和范围
    center = all_points.mean(axis=0)
    centered_points = all_points - center
    
    # PCA找主要方向
    cov_matrix = np.cov(centered_points.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx]
    
    # 在主平面上生成网格
    extent = 20.0
    u = np.linspace(-extent/2, extent/2, grid_size)
    v = np.linspace(-extent/2, extent/2, grid_size)
    U, V = np.meshgrid(u, v)
    
    # 生成弯曲表面
    surface_points = np.zeros((grid_size, grid_size, 3))
    
    for i in range(grid_size):
        for j in range(grid_size):
            # 基础平面位置
            base_pos = U[i,j] * eigenvecs[:, 0] + V[i,j] * eigenvecs[:, 1]
            
            # 添加弯曲（基于距离的正弦函数）
            dist_from_center = np.sqrt(U[i,j]**2 + V[i,j]**2)
            curvature = 3.0 * np.sin(dist_from_center * 0.3) * np.exp(-dist_from_center * 0.1)
            
            # 最终位置
            surface_points[i, j] = center + base_pos + curvature * eigenvecs[:, 2]
    
    return surface_points

def batch_validation_demo():
    """批量验证演示"""
    tree_files = glob.glob('tree_*.json')
    if len(tree_files) < 2:
        print("批量验证需要至少2个血管树文件")
        return
    
    print(f"批量验证 {len(tree_files)} 个血管树...")
    
    results_summary = []
    
    for i, tree_file in enumerate(tree_files):
        print(f"\n处理文件 {i+1}/{len(tree_files)}: {tree_file}")
        
        # 生成演示曲面
        demo_surface = create_demo_surface(tree_file, grid_size=16)
        
        # 分析质量
        metrics = analyze_surface_quality(tree_file, demo_surface)
        
        # 保存可视化
        base_name = tree_file.replace('.json', '')
        visualize_generated_surface(
            tree_file, 
            demo_surface, 
            save_path=f"batch_{base_name}_validation.png"
        )
        
        # 收集关键指标
        summary = {
            'file': tree_file,
            'center_distance': metrics['center_distance'],
            'mean_fit_distance': metrics['min_distance_stats']['mean'],
            'coverage_ratio': metrics['coverage_ratio'].mean()  # 如果是数组的话
        }
        results_summary.append(summary)
        
        print(f"  中心距离: {summary['center_distance']:.4f}")
        print(f"  拟合距离: {summary['mean_fit_distance']:.4f}")
    
    # 汇总报告
    print("\n" + "="*50)
    print("批量验证汇总报告")
    print("="*50)
    
    center_distances = [r['center_distance'] for r in results_summary]
    fit_distances = [r['mean_fit_distance'] for r in results_summary]
    
    print(f"平均中心距离: {np.mean(center_distances):.4f} ± {np.std(center_distances):.4f}")
    print(f"平均拟合距离: {np.mean(fit_distances):.4f} ± {np.std(fit_distances):.4f}")
    
    return results_summary

def ideal_surface_demo():
    """理想曲面演示"""
    tree_files = glob.glob('tree_*.json')
    if not tree_files:
        print("未找到血管树文件")
        return
    
    print("理想曲面验证演示")
    print("=" * 50)
    print("这个功能将基于以下数据生成理想曲面:")
    print("- 血管主干点")
    print("- 两个分支对应点的中点")
    print("- PCA主成分分析确定的主平面")
    print("- 轻微的距离相关弯曲")
    
    test_file = tree_files[0]
    print(f"\n使用血管树文件: {test_file}")
    
    # 生成和验证理想曲面
    ideal_surface = validate_ideal_surface_only(
        test_file, 
        grid_size=24,  # 中等分辨率
        save_prefix="ideal_demo"
    )
    
    print("\n理想曲面演示完成！生成的文件:")
    print("- ideal_demo_visualization.png: 理想曲面可视化")
    print("- ideal_demo_metrics.json: 理想曲面质量指标")
    
    return ideal_surface

def ideal_vs_generated_demo():
    """理想曲面与生成曲面对比演示"""
    tree_files = glob.glob('tree_*.json')
    if not tree_files:
        print("未找到血管树文件")
        return
    
    print("理想曲面 vs 生成曲面对比演示")
    print("=" * 50)
    
    test_file = tree_files[0]
    grid_size = 20  # 较小网格加快演示
    
    # 1. 生成理想曲面
    print("步骤1: 生成理想曲面...")
    ideal_surface = generate_ideal_surface_from_tree(test_file, grid_size)
    
    # 2. 训练轻量级模型并生成曲面
    print("步骤2: 训练轻量级模型...")
    model, betas = train_tree_diffusion(
        [test_file], 
        epochs=500,  # 快速训练用于演示
        batch_size=1, 
        device='cpu', 
        grid_size=grid_size
    )
    
    print("步骤3: 生成扩散模型曲面...")
    generated_surface = denoise_with_tree(
        test_file, 
        model, 
        betas, 
        device='cpu', 
        grid_size=grid_size
    )
    
    # 3. 对比分析
    print("步骤4: 对比分析...")
    ideal_surface, diff_metrics = compare_ideal_vs_generated(
        test_file, 
        generated_surface, 
        save_prefix="comparison_demo"
    )
    
    print("\n对比演示完成！生成的文件:")
    print("- comparison_demo_ideal_vs_generated.png: 四联图对比")
    print("- comparison_demo_ideal_comparison_metrics.json: 差异指标")
    
    return ideal_surface, generated_surface, diff_metrics

def batch_ideal_surfaces_demo():
    """批量理想曲面生成演示"""
    tree_files = glob.glob('tree_*.json')
    if len(tree_files) < 2:
        print("批量理想曲面演示需要至少2个血管树文件")
        return
    
    print(f"批量生成 {len(tree_files)} 个理想曲面...")
    
    ideal_surfaces = []
    quality_summary = []
    
    for i, tree_file in enumerate(tree_files):
        print(f"\n处理文件 {i+1}/{len(tree_files)}: {tree_file}")
        
        # 生成理想曲面
        ideal_surface = generate_ideal_surface_from_tree(tree_file, grid_size=16)
        ideal_surfaces.append(ideal_surface)
        
        # 分析质量
        metrics = analyze_surface_quality(tree_file, ideal_surface)
        
        # 保存可视化
        base_name = tree_file.replace('.json', '')
        visualize_generated_surface(
            tree_file, 
            ideal_surface, 
            save_path=f"ideal_{base_name}.png"
        )
        
        # 收集质量指标
        quality_summary.append({
            'file': tree_file,
            'center_distance': metrics['center_distance'],
            'mean_fit_distance': metrics['min_distance_stats']['mean'],
            'coverage_ratio': metrics['coverage_ratio'].mean() if hasattr(metrics['coverage_ratio'], 'mean') else metrics['coverage_ratio']
        })
        
        print(f"  中心距离: {quality_summary[-1]['center_distance']:.4f}")
        print(f"  拟合距离: {quality_summary[-1]['mean_fit_distance']:.4f}")
    
    # 汇总报告
    print("\n" + "="*50)
    print("批量理想曲面质量汇总")
    print("="*50)
    
    center_distances = [q['center_distance'] for q in quality_summary]
    fit_distances = [q['mean_fit_distance'] for q in quality_summary]
    
    print(f"理想曲面数量: {len(quality_summary)}")
    print(f"平均中心距离: {np.mean(center_distances):.4f} ± {np.std(center_distances):.4f}")
    print(f"平均拟合距离: {np.mean(fit_distances):.4f} ± {np.std(fit_distances):.4f}")
    
    print(f"\n每个理想曲面的可视化已保存为 ideal_tree_*.png")
    
    return ideal_surfaces, quality_summary

if __name__ == "__main__":
    print("血管曲面验证演示菜单:")
    print("1. 快速验证演示")
    print("2. 交互式验证")
    print("3. 批量验证演示")
    print("4. 理想曲面演示")
    print("5. 理想曲面 vs 生成曲面对比")
    print("6. 批量理想曲面生成")
    
    try:
        choice = input("请选择 (1-6): ").strip()
        
        if choice == '1':
            quick_validation_demo()
        elif choice == '2':
            interactive_validation()
        elif choice == '3':
            batch_validation_demo()
        elif choice == '4':
            ideal_surface_demo()
        elif choice == '5':
            ideal_vs_generated_demo()
        elif choice == '6':
            batch_ideal_surfaces_demo()
        else:
            print("运行快速验证演示...")
            quick_validation_demo()
            
    except KeyboardInterrupt:
        print("\n演示中断")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("建议检查血管树文件是否存在且格式正确") 