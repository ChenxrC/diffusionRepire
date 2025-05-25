import numpy as np
import open3d as o3d
import json
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_points_from_branch(branch: Dict[str, Any]) -> np.ndarray:
    """从分支中提取所有点
    
    Args:
        branch: 分支数据字典
    
    Returns:
        分支上的所有点
    """
    points = np.array(branch["points"])
    for child in branch["children"]:
        child_points = extract_points_from_branch(child)
        points = np.vstack((points, child_points))
    return points

def load_tree_pointcloud(filename: str = "tree_pointcloud.json") -> np.ndarray:
    """加载树形点云数据
    
    Args:
        filename: JSON文件名
    
    Returns:
        点云数据数组
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    all_points = []
    for branch in data["branches"]:
        points = extract_points_from_branch(branch)
        all_points.append(points)
    
    return np.vstack(all_points)

def load_forest_pointcloud(filename: str = "forest_pointcloud.json") -> np.ndarray:
    """加载森林点云数据
    
    Args:
        filename: JSON文件名
    
    Returns:
        点云数据数组
    """
    with open(filename, 'r') as f:
        forest_data = json.load(f)
    
    all_points = []
    for tree in forest_data:
        for branch in tree["branches"]:
            points = extract_points_from_branch(branch)
            all_points.append(points)
    
    return np.vstack(all_points)

def create_plane_points(center: np.ndarray, normal: np.ndarray, size: float = 2.0, num_points: int = 100) -> np.ndarray:
    """创建平面点云
    
    Args:
        center: 平面中心点
        normal: 平面法向量
        size: 平面大小
        num_points: 平面上的点数
    
    Returns:
        平面上的点云数据
    """
    # 创建两个正交向量
    v1 = np.array([1, 0, 0])
    if abs(np.dot(v1, normal)) > 0.9:
        v1 = np.array([0, 1, 0])
    v1 = v1 - np.dot(v1, normal) * normal
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    
    # 生成平面上的点
    points = []
    for i in range(num_points):
        t1 = np.random.uniform(-size/2, size/2)
        t2 = np.random.uniform(-size/2, size/2)
        point = center + t1 * v1 + t2 * v2
        points.append(point)
    
    return np.array(points)

def visualize_pointcloud(points: np.ndarray, planes: List[np.ndarray] = None):
    """可视化点云和平面
    
    Args:
        points: 点云数据
        planes: 平面点云数据列表
    """
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置点云颜色为绿色
    colors = np.ones((len(points), 3)) * np.array([0, 1, 0])
    # 将第一个点设置为蓝色
    colors[0] = np.array([0, 0, 1])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 添加点云
    vis.add_geometry(pcd)
    
    # 添加平面（如果存在）
    if planes:
        for i, plane_points in enumerate(planes):
            plane_pcd = o3d.geometry.PointCloud()
            plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
            # 设置平面颜色为红色
            plane_pcd.paint_uniform_color([1, 0, 0])
            vis.add_geometry(plane_pcd)
    
    # 设置视角
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, 1, 0])
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def find_max_points_branches(tree_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """找到主干和两个最大分支的点
    
    Args:
        tree_data: 树形数据
    
    Returns:
        主干点云，第一个分支点云，第二个分支点云
    """
    # 获取主干点
    trunk_points = np.array(tree_data["branches"][0]["points"])
    
    # 获取所有一级分支
    first_level_branches = tree_data["branches"][0]["children"]
    
    # 按点数排序分支
    sorted_branches = sorted(first_level_branches, 
                           key=lambda x: len(x["points"]), 
                           reverse=True)
    
    # 获取点数最多的两个分支
    branch1_points = np.array(sorted_branches[0]["points"])
    branch2_points = np.array(sorted_branches[1]["points"])
    
    return trunk_points, branch1_points, branch2_points

def calculate_plane_equation(points: np.ndarray, normal_constraint: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """计算平面的法向量和常数项
    
    Args:
        points: 平面上的点
        normal_constraint: 平面的法向量约束（如果提供）
    
    Returns:
        平面法向量和常数项
    """
    # 使用SVD计算平面方程
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    if normal_constraint is not None:
        # 如果提供了法向量约束，使用它作为平面的法向量
        normal = normal_constraint
    else:
        # 否则使用SVD计算最佳拟合平面
        U, S, Vh = np.linalg.svd(centered_points)
        normal = Vh[2, :]  # 最小奇异值对应的右奇异向量
    
    d = -np.dot(normal, centroid)
    return normal, d

def calculate_perpendicular_plane(points: np.ndarray, normal_constraint: np.ndarray) -> Tuple[np.ndarray, float]:
    """计算垂直于给定法向量的平面
    
    Args:
        points: 平面上的点
        normal_constraint: 需要垂直的法向量
    
    Returns:
        平面法向量和常数项
    """
    # 使用SVD计算平面方程
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # 使用SVD计算最佳拟合平面
    U, S, Vh = np.linalg.svd(centered_points)
    
    # 获取前两个主方向
    v1 = Vh[0, :]  # 第一主方向
    v2 = Vh[1, :]  # 第二主方向
    
    # 计算垂直于约束法向量的方向
    # 首先尝试使用第一主方向
    perpendicular = np.cross(normal_constraint, v1)
    if np.linalg.norm(perpendicular) < 1e-10:  # 如果叉积接近零向量
        perpendicular = np.cross(normal_constraint, v2)
    
    # 确保法向量是单位向量
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    
    # 验证垂直性
    dot_product = np.abs(np.dot(perpendicular, normal_constraint))
    if dot_product > 1e-10:
        # 如果不够垂直，使用Gram-Schmidt正交化
        perpendicular = perpendicular - np.dot(perpendicular, normal_constraint) * normal_constraint
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
    
    # 计算平面方程
    d = -np.dot(perpendicular, centroid)
    return perpendicular, d

def plot_points_and_planes(trunk_points: np.ndarray, 
                          branch1_points: np.ndarray, 
                          branch2_points: np.ndarray):
    """绘制点和两个垂直平面
    
    Args:
        trunk_points: 主干点云
        branch1_points: 第一个分支点云
        branch2_points: 第二个分支点云
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点
    trunk_scatter = ax.scatter(trunk_points[:, 0], trunk_points[:, 1], trunk_points[:, 2], 
                             c='blue', label='Trunk', s=30)
    branch1_scatter = ax.scatter(branch1_points[:, 0], branch1_points[:, 1], branch1_points[:, 2], 
                               c='green', label='Branch 1', s=30)
    branch2_scatter = ax.scatter(branch2_points[:, 0], branch2_points[:, 1], branch2_points[:, 2], 
                               c='red', label='Branch 2', s=30)
    
    # 计算第一个平面（使用两个分支的点）
    branch_points = np.vstack([branch1_points, branch2_points])
    normal1, d1 = calculate_plane_equation(branch_points)
    
    # 计算第二个平面（使用主干的点，且垂直于第一个平面）
    normal2, d2 = calculate_perpendicular_plane(trunk_points, normal1)
    
    # 创建平面网格
    x = np.linspace(min(branch_points[:, 0]), max(branch_points[:, 0]), 20)
    y = np.linspace(min(branch_points[:, 1]), max(branch_points[:, 1]), 20)
    X, Y = np.meshgrid(x, y)
    
    # 绘制第一个平面
    Z1 = (-normal1[0] * X - normal1[1] * Y - d1) / normal1[2]
    branch_plane = ax.plot_surface(X, Y, Z1, alpha=0.2, color='green')
    
    # 绘制第二个平面
    Z2 = (-normal2[0] * X - normal2[1] * Y - d2) / normal2[2]
    trunk_plane = ax.plot_surface(X, Y, Z2, alpha=0.2, color='blue')
    
    # 绘制法向量
    center1 = np.mean(branch_points, axis=0)
    center2 = np.mean(trunk_points, axis=0)
    
    # 绘制第一个平面的法向量
    branch_normal = ax.quiver(center1[0], center1[1], center1[2],
                            normal1[0], normal1[1], normal1[2],
                            length=5, color='green', arrow_length_ratio=0.3)
    
    # 绘制第二个平面的法向量
    trunk_normal = ax.quiver(center2[0], center2[1], center2[2],
                           normal2[0], normal2[1], normal2[2],
                           length=5, color='blue', arrow_length_ratio=0.3)
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Trunk',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Branch 1',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Branch 2',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='green', label='Branch Plane',
               alpha=0.2, linewidth=10),
        Line2D([0], [0], color='blue', label='Trunk Plane',
               alpha=0.2, linewidth=10)
    ]
    ax.legend(handles=legend_elements)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 打印平面方程和法向量点积（应该接近0）
    print("第一个平面方程（分支平面）:")
    print(f"{normal1[0]:.2f}x + {normal1[1]:.2f}y + {normal1[2]:.2f}z + {d1:.2f} = 0")
    print("\n第二个平面方程（主干平面）:")
    print(f"{normal2[0]:.2f}x + {normal2[1]:.2f}y + {normal2[2]:.2f}z + {d2:.2f} = 0")
    print("\n两个法向量的点积（应该接近0）:")
    print(f"{np.dot(normal1, normal2):.6f}")
    
    plt.show()

if __name__ == "__main__":
    # 加载树形点云数据
    try:
        with open('tree_pointcloud.json', 'r') as f:
            tree_data = json.load(f)
        print("成功加载单棵树的数据")
    except FileNotFoundError:
        try:
            with open('forest_pointcloud.json', 'r') as f:
                forest_data = json.load(f)
                tree_data = forest_data[0]  # 使用第一棵树
            print("成功加载森林数据")
        except FileNotFoundError:
            print("未找到点云数据文件")
            exit(1)
    
    # 找到主干和最大分支的点
    trunk_points, branch1_points, branch2_points = find_max_points_branches(tree_data)
    
    # 绘制点和平面
    plot_points_and_planes(trunk_points, branch1_points, branch2_points) 