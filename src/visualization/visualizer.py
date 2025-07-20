"""
可视化工具模块
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Optional, Tuple


def demo_separation_methods():
    """
    演示分离方法
    """
    print("=== 分离方法演示 ===")
    
    # 生成示例数据
    np.random.seed(42)
    
    # 组1：球形分布
    n1 = 100
    r1 = 2.0
    theta1 = np.random.uniform(0, 2*np.pi, n1)
    phi1 = np.random.uniform(0, np.pi, n1)
    x1 = r1 * np.sin(phi1) * np.cos(theta1) + np.random.normal(0, 0.3, n1)
    y1 = r1 * np.sin(phi1) * np.sin(theta1) + np.random.normal(0, 0.3, n1)
    z1 = r1 * np.cos(phi1) + np.random.normal(0, 0.3, n1)
    points_group1 = np.column_stack([x1, y1, z1])
    
    # 组2：椭球形分布
    n2 = 100
    r2 = 3.0
    theta2 = np.random.uniform(0, 2*np.pi, n2)
    phi2 = np.random.uniform(0, np.pi, n2)
    x2 = r2 * np.sin(phi2) * np.cos(theta2) + np.random.normal(0, 0.3, n2)
    y2 = r2 * np.sin(phi2) * np.sin(theta2) + np.random.normal(0, 0.3, n2)
    z2 = r2 * np.cos(phi2) + np.random.normal(0, 0.3, n2)
    points_group2 = np.column_stack([x2, y2, z2])
    points_group2 += np.array([4, 2, 1])  # 移动到不同位置
    
    print(f"生成示例数据:")
    print(f"  组1点数: {len(points_group1)}")
    print(f"  组2点数: {len(points_group2)}")
    
    # 可视化原始数据
    visualize_point_groups(points_group1, points_group2, "原始点云数据")
    
    # 尝试不同的分离方法
    try:
        from ..separation import create_svm_separation_surface, create_lda_separation_surface, create_kmeans_separation_surface
        
        # SVM分离
        print("\n1. SVM分离...")
        svm_surface, svm_model, svm_info = create_svm_separation_surface(
            points_group1, points_group2, kernel='rbf', C=1.0
        )
        print(f"   SVM准确率: {svm_info['accuracy']:.3f}")
        print(f"   SVM分离度: {svm_info['separation_degree']:.3f}")
        
        # LDA分离
        print("\n2. LDA分离...")
        lda_surface, lda_model, lda_info = create_lda_separation_surface(
            points_group1, points_group2
        )
        print(f"   LDA准确率: {lda_info['accuracy']:.3f}")
        print(f"   LDA分离度: {lda_info['separation_degree']:.3f}")
        
        # K-means分离
        print("\n3. K-means分离...")
        kmeans_surface, kmeans_model, kmeans_info = create_kmeans_separation_surface(
            points_group1, points_group2
        )
        print(f"   K-means准确率: {kmeans_info['accuracy']:.3f}")
        print(f"   K-means分离度: {kmeans_info['separation_degree']:.3f}")
        
        # 可视化所有分离结果
        visualize_all_separation_results(
            points_group1, points_group2,
            {
                'svm': {'surface': svm_surface, 'info': svm_info},
                'lda': {'surface': lda_surface, 'info': lda_info},
                'kmeans': {'surface': kmeans_surface, 'info': kmeans_info}
            }
        )
        
    except ImportError as e:
        print(f"⚠️  分离模块导入失败: {e}")
        print("   请确保已安装scikit-learn: pip install scikit-learn")


def visualize_point_groups(points_group1: np.ndarray, points_group2: np.ndarray, title: str = "点云数据"):
    """
    可视化两组点云
    
    Args:
        points_group1: 第一组点
        points_group2: 第二组点
        title: 图表标题
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制两组点
    ax.scatter(points_group1[:, 0], points_group1[:, 1], points_group1[:, 2], 
              c='red', s=20, alpha=0.7, label='组1')
    ax.scatter(points_group2[:, 0], points_group2[:, 1], points_group2[:, 2], 
              c='blue', s=20, alpha=0.7, label='组2')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # 保存图片
    os.makedirs("visualization_results", exist_ok=True)
    save_path = f"visualization_results/{title.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存: {save_path}")


def visualize_all_separation_results(points_group1: np.ndarray, points_group2: np.ndarray, 
                                   results: dict):
    """
    可视化所有分离方法的结果
    
    Args:
        points_group1: 第一组点
        points_group2: 第二组点
        results: 分离结果字典
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 创建子图
    plot_idx = 1
    for method_name, result in results.items():
        ax = fig.add_subplot(2, 3, plot_idx, projection='3d')
        
        # 绘制原始点
        ax.scatter(points_group1[:, 0], points_group1[:, 1], points_group1[:, 2], 
                  c='red', s=20, alpha=0.7, label='组1')
        ax.scatter(points_group2[:, 0], points_group2[:, 1], points_group2[:, 2], 
                  c='blue', s=20, alpha=0.7, label='组2')
        
        # 绘制分离曲面
        surface = result['surface']
        ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], 
                  c='green', s=10, alpha=0.5, label='分离曲面')
        
        # 添加统计信息
        info = result['info']
        stats_text = f"准确率: {info['accuracy']:.3f}\n分离度: {info['separation_degree']:.3f}"
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{method_name.upper()} 分离结果')
        ax.legend()
        
        plot_idx += 1
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs("visualization_results", exist_ok=True)
    save_path = "visualization_results/all_separation_methods.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"所有分离方法结果已保存: {save_path}")


def visualize_3d_surface(surface_points: np.ndarray, title: str = "3D曲面", 
                        save_path: Optional[str] = None, interactive: bool = True):
    """
    可视化3D曲面
    
    Args:
        surface_points: 曲面点 (N, 3)
        title: 图表标题
        save_path: 保存路径
        interactive: 是否交互式显示
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面点
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], 
              c='green', s=20, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if interactive:
        plt.show()
    else:
        plt.close()


def create_comparison_plot(data_dict: dict, title: str = "方法比较", 
                          save_path: Optional[str] = None):
    """
    创建比较图表
    
    Args:
        data_dict: 数据字典，格式为 {方法名: 指标值}
        title: 图表标题
        save_path: 保存路径
    """
    methods = list(data_dict.keys())
    values = list(data_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, values, alpha=0.7)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    ax.set_title(title)
    ax.set_ylabel('指标值')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show() 