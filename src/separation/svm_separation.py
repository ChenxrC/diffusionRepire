"""
SVM分离方法模块
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Tuple, Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def create_svm_separation_surface(points_group1: np.ndarray, points_group2: np.ndarray, 
                                grid_size: int = 32, plane_size: float = 10.0, 
                                kernel: str = 'rbf', C: float = 1.0) -> Tuple[np.ndarray, SVC, Dict[str, Any]]:
    """
    使用SVM创建分离曲面
    
    Args:
        points_group1: 第一组点 (N1, 3)
        points_group2: 第二组点 (N2, 3)
        grid_size: 网格大小
        plane_size: 平面大小
        kernel: SVM核函数 ('linear', 'rbf', 'poly')
        C: SVM正则化参数
    
    Returns:
        separation_surface: 分离曲面点 (grid_size*grid_size, 3)
        svm_model: 训练好的SVM模型
        separation_info: 分离信息字典
    """
    # 准备训练数据
    X = np.vstack([points_group1, points_group2])
    y = np.hstack([np.zeros(len(points_group1)), np.ones(len(points_group2))])
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练SVM模型
    svm_model = SVC(kernel=kernel, C=C, probability=True)
    svm_model.fit(X_scaled, y)
    
    # 预测训练集准确率
    y_pred = svm_model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    # 计算分离度
    separation_degree = _calculate_separation_degree(points_group1, points_group2, svm_model, scaler)
    
    # 生成分离曲面
    separation_surface = _generate_separation_surface(
        points_group1, points_group2, svm_model, scaler, grid_size, plane_size, kernel
    )
    
    # 收集分离信息
    separation_info = {
        'accuracy': accuracy,
        'separation_degree': separation_degree,
        'support_vectors_count': len(svm_model.support_vectors_),
        'kernel': kernel,
        'C': C
    }
    
    return separation_surface, svm_model, separation_info


def create_lda_separation_surface(points_group1: np.ndarray, points_group2: np.ndarray,
                                grid_size: int = 32, plane_size: float = 10.0) -> Tuple[np.ndarray, LinearDiscriminantAnalysis, Dict[str, Any]]:
    """
    使用LDA创建分离曲面
    
    Args:
        points_group1: 第一组点 (N1, 3)
        points_group2: 第二组点 (N2, 3)
        grid_size: 网格大小
        plane_size: 平面大小
    
    Returns:
        separation_surface: 分离曲面点
        lda_model: 训练好的LDA模型
        separation_info: 分离信息字典
    """
    # 准备训练数据
    X = np.vstack([points_group1, points_group2])
    y = np.hstack([np.zeros(len(points_group1)), np.ones(len(points_group2))])
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练LDA模型
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_scaled, y)
    
    # 预测训练集准确率
    y_pred = lda_model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    # 计算分离度
    separation_degree = _calculate_separation_degree(points_group1, points_group2, lda_model, scaler)
    
    # 生成分离曲面
    separation_surface = _generate_lda_surface(
        points_group1, points_group2, lda_model, scaler, grid_size, plane_size
    )
    
    # 收集分离信息
    separation_info = {
        'accuracy': accuracy,
        'separation_degree': separation_degree,
        'method': 'LDA'
    }
    
    return separation_surface, lda_model, separation_info


def create_kmeans_separation_surface(points_group1: np.ndarray, points_group2: np.ndarray,
                                   grid_size: int = 32, plane_size: float = 10.0) -> Tuple[np.ndarray, KMeans, Dict[str, Any]]:
    """
    使用K-means创建分离曲面
    
    Args:
        points_group1: 第一组点 (N1, 3)
        points_group2: 第二组点 (N2, 3)
        grid_size: 网格大小
        plane_size: 平面大小
    
    Returns:
        separation_surface: 分离曲面点
        kmeans_model: 训练好的K-means模型
        separation_info: 分离信息字典
    """
    # 准备训练数据
    X = np.vstack([points_group1, points_group2])
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练K-means模型
    kmeans_model = KMeans(n_clusters=2, random_state=42)
    y_pred = kmeans_model.fit_predict(X_scaled)
    
    # 计算准确率（需要对齐标签）
    y_true = np.hstack([np.zeros(len(points_group1)), np.ones(len(points_group2))])
    
    # 对齐聚类标签
    if np.mean(y_pred[:len(points_group1)]) > np.mean(y_pred[len(points_group1):]):
        y_pred = 1 - y_pred
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算分离度
    separation_degree = _calculate_separation_degree(points_group1, points_group2, kmeans_model, scaler)
    
    # 生成分离曲面
    separation_surface = _generate_kmeans_surface(
        points_group1, points_group2, kmeans_model, scaler, grid_size, plane_size
    )
    
    # 收集分离信息
    separation_info = {
        'accuracy': accuracy,
        'separation_degree': separation_degree,
        'method': 'K-means'
    }
    
    return separation_surface, kmeans_model, separation_info


def _calculate_separation_degree(points_group1: np.ndarray, points_group2: np.ndarray, 
                               model: Any, scaler: StandardScaler) -> float:
    """
    计算分离度
    
    Args:
        points_group1: 第一组点
        points_group2: 第二组点
        model: 训练好的模型
        scaler: 标准化器
    
    Returns:
        separation_degree: 分离度
    """
    # 计算两组点的中心
    center1 = np.mean(points_group1, axis=0)
    center2 = np.mean(points_group2, axis=0)
    
    # 计算中心距离
    center_distance = np.linalg.norm(center2 - center1)
    
    # 计算组内平均距离
    def group_avg_distance(points):
        if len(points) < 2:
            return 0
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                distances.append(np.linalg.norm(points[i] - points[j]))
        return np.mean(distances) if distances else 0
    
    avg_distance1 = group_avg_distance(points_group1)
    avg_distance2 = group_avg_distance(points_group2)
    avg_group_distance = (avg_distance1 + avg_distance2) / 2
    
    # 分离度 = 中心距离 / 平均组内距离
    separation_degree = center_distance / (avg_group_distance + 1e-8)
    
    return separation_degree


def _generate_separation_surface(points_group1: np.ndarray, points_group2: np.ndarray,
                               model: SVC, scaler: StandardScaler, grid_size: int, 
                               plane_size: float, kernel: str) -> np.ndarray:
    """
    生成SVM分离曲面
    
    Args:
        points_group1: 第一组点
        points_group2: 第二组点
        model: SVM模型
        scaler: 标准化器
        grid_size: 网格大小
        plane_size: 平面大小
        kernel: 核函数类型
    
    Returns:
        separation_surface: 分离曲面点
    """
    # 计算两组点的边界框
    all_points = np.vstack([points_group1, points_group2])
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    
    # 扩展边界框
    center = (min_coords + max_coords) / 2
    extent = (max_coords - min_coords) / 2
    extent = np.maximum(extent, plane_size / 2)
    
    # 生成网格
    x = np.linspace(center[0] - extent[0], center[0] + extent[0], grid_size)
    y = np.linspace(center[1] - extent[1], center[1] + extent[1], grid_size)
    z = np.linspace(center[2] - extent[2], center[2] + extent[2], grid_size)
    
    # 创建网格点
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # 标准化网格点
    grid_points_scaled = scaler.transform(grid_points)
    
    # 预测每个网格点的类别
    predictions = model.predict(grid_points_scaled)
    
    # 找到决策边界附近的点
    if hasattr(model, 'decision_function'):
        decision_values = model.decision_function(grid_points_scaled)
        boundary_mask = np.abs(decision_values) < 0.1  # 决策边界附近的点
    else:
        # 对于没有decision_function的模型，使用概率
        probabilities = model.predict_proba(grid_points_scaled)
        boundary_mask = np.abs(probabilities[:, 0] - probabilities[:, 1]) < 0.1
    
    # 提取边界点
    boundary_points = grid_points[boundary_mask]
    
    # 如果边界点太少，使用所有预测为边界类的点
    if len(boundary_points) < grid_size * grid_size // 4:
        # 使用概率接近0.5的点
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(grid_points_scaled)
            boundary_mask = np.abs(probabilities[:, 0] - 0.5) < 0.2
            boundary_points = grid_points[boundary_mask]
    
    # 如果仍然太少，随机采样
    if len(boundary_points) < grid_size * grid_size // 8:
        n_samples = grid_size * grid_size
        indices = np.random.choice(len(grid_points), n_samples, replace=False)
        boundary_points = grid_points[indices]
    
    return boundary_points


def _generate_lda_surface(points_group1: np.ndarray, points_group2: np.ndarray,
                         model: LinearDiscriminantAnalysis, scaler: StandardScaler,
                         grid_size: int, plane_size: float) -> np.ndarray:
    """
    生成LDA分离曲面
    """
    # 使用与SVM相同的方法
    return _generate_separation_surface(points_group1, points_group2, model, scaler, grid_size, plane_size, 'linear')


def _generate_kmeans_surface(points_group1: np.ndarray, points_group2: np.ndarray,
                           model: KMeans, scaler: StandardScaler,
                           grid_size: int, plane_size: float) -> np.ndarray:
    """
    生成K-means分离曲面
    """
    # 使用与SVM相同的方法
    return _generate_separation_surface(points_group1, points_group2, model, scaler, grid_size, plane_size, 'linear')


def compare_separation_methods_for_vascular_data(tree_json: str):
    """
    比较血管数据的分离方法
    
    Args:
        tree_json: 血管树JSON文件路径
    """
    print(f"=== 血管数据分离方法比较 ===")
    print(f"数据文件: {tree_json}")
    
    try:
        import json
        from ..utils.vascular_utils import find_max_points_branches
        
        # 加载血管数据
        with open(tree_json, 'r') as f:
            tree_data = json.load(f)
        
        # 提取主干和分支点
        trunk_pts, br1_pts, br2_pts = find_max_points_branches(tree_data)
        
        if len(trunk_pts) == 0 or len(br1_pts) == 0 or len(br2_pts) == 0:
            print("⚠️  无法提取有效的血管点数据")
            return
        
        print(f"提取的血管点:")
        print(f"  主干点数: {len(trunk_pts)}")
        print(f"  分支1点数: {len(br1_pts)}")
        print(f"  分支2点数: {len(br2_pts)}")
        
        # 比较分离方法
        results = {}
        
        # SVM分离
        print("\n1. SVM分离...")
        svm_surface, svm_model, svm_info = create_svm_separation_surface(
            trunk_pts, np.vstack([br1_pts, br2_pts]), kernel='rbf', C=1.0
        )
        results['svm'] = {'surface': svm_surface, 'info': svm_info}
        print(f"   SVM准确率: {svm_info['accuracy']:.3f}")
        print(f"   SVM分离度: {svm_info['separation_degree']:.3f}")
        
        # LDA分离
        print("\n2. LDA分离...")
        lda_surface, lda_model, lda_info = create_lda_separation_surface(
            trunk_pts, np.vstack([br1_pts, br2_pts])
        )
        results['lda'] = {'surface': lda_surface, 'info': lda_info}
        print(f"   LDA准确率: {lda_info['accuracy']:.3f}")
        print(f"   LDA分离度: {lda_info['separation_degree']:.3f}")
        
        # K-means分离
        print("\n3. K-means分离...")
        kmeans_surface, kmeans_model, kmeans_info = create_kmeans_separation_surface(
            trunk_pts, np.vstack([br1_pts, br2_pts])
        )
        results['kmeans'] = {'surface': kmeans_surface, 'info': kmeans_info}
        print(f"   K-means准确率: {kmeans_info['accuracy']:.3f}")
        print(f"   K-means分离度: {kmeans_info['separation_degree']:.3f}")
        
        # 可视化结果
        from ..visualization.visualizer import visualize_all_separation_results
        visualize_all_separation_results(
            trunk_pts, np.vstack([br1_pts, br2_pts]), results
        )
        
        # 创建比较图表
        accuracies = {method: result['info']['accuracy'] for method, result in results.items()}
        separation_degrees = {method: result['info']['separation_degree'] for method, result in results.items()}
        
        from ..visualization.visualizer import create_comparison_plot
        create_comparison_plot(accuracies, "分离方法准确率比较", "visualization_results/accuracy_comparison.png")
        create_comparison_plot(separation_degrees, "分离方法分离度比较", "visualization_results/separation_degree_comparison.png")
        
        print("\n✅ 血管数据分离方法比较完成!")
        
    except Exception as e:
        print(f"❌ 处理血管数据时出错: {e}")
        print("请确保数据文件格式正确且包含有效的血管点数据") 