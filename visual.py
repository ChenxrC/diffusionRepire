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

# ----------  find_max_points_branches utility (ensure defined) ----------
try:
    find_max_points_branches  # type: ignore
except NameError:
    def find_max_points_branches(tree_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """遍历 tree_data 取得主干与点数最多的两条一级分支点云
        返回 (trunk_pts, branch1_pts, branch2_pts)
        若一级分支不足 2 条则返回空数组占位"""
        trunk_points = np.array(tree_data["branches"][0]["points"], dtype=np.float32)
        branches = tree_data["branches"][0].get("children", [])
        if len(branches) < 2:
            # fallback: 直接按 children 全部或自己复制
            if len(branches) == 1:
                branch1_pts = np.array(branches[0]["points"], dtype=np.float32)
                branch2_pts = branch1_pts.copy()
            else:
                branch1_pts = branch2_pts = trunk_points.copy()
            return trunk_points, branch1_pts, branch2_pts
        # sort by number of points
        branches_sorted = sorted(branches, key=lambda b: len(b["points"]), reverse=True)
        branch1_pts = np.array(branches_sorted[0]["points"], dtype=np.float32)
        branch2_pts = np.array(branches_sorted[1]["points"], dtype=np.float32)
        return trunk_points, branch1_pts, branch2_pts

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
    """绘制主干与两条分支点云、对应的两个互相垂直平面以及法向量
    
    trunk_points   : 主干点云 (N,3)
    branch1_points : 第一分支点云 (M,3)
    branch2_points : 第二分支点云 (K,3)
    """
    # 准备 Matplotlib 3D 轴
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点
    ax.scatter(*trunk_points.T,  c='blue',  s=30, label='Trunk')
    ax.scatter(*branch1_points.T, c='green', s=30, label='Branch 1')
    ax.scatter(*branch2_points.T, c='red',   s=30, label='Branch 2')

    # ---------- 1. 计算两个平面 ---------- #
    branch_points = np.vstack([branch1_points, branch2_points])
    normal1, _ = calculate_plane_equation(branch_points)            # 分支平面法向量
    normal1 /= np.linalg.norm(normal1)

    normal2, _ = calculate_perpendicular_plane(trunk_points, normal1) # 主干平面法向量 (垂直 normal1)
    normal2 /= np.linalg.norm(normal2)

    # ---------- 2. 生成平面网格 ---------- #
    def plane_mesh(center: np.ndarray, normal: np.ndarray,
                   plane_size: float = 30.0, density: int = 20):
        """基于中心点与法向量生成矩形网格 (X,Y,Z)"""
        # 找到在该平面内的两个正交方向 v1,v2
        helper = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(np.dot(helper, normal)), 1.0, atol=1e-3):
            helper = np.array([0.0, 1.0, 0.0])
        v1 = np.cross(normal, helper)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 /= np.linalg.norm(v2)

        grid = np.linspace(-plane_size/2, plane_size/2, density)
        u, v = np.meshgrid(grid, grid)
        pts = center + u[..., None]*v1 + v[..., None]*v2
        return pts[...,0], pts[...,1], pts[...,2]

    center1 = branch_points.mean(axis=0)
    center2 = trunk_points.mean(axis=0)

    X1, Y1, Z1 = plane_mesh(center1, normal1)
    X2, Y2, Z2 = plane_mesh(center2, normal2)

    ax.plot_surface(X1, Y1, Z1, alpha=0.25, color='lightgreen')
    ax.plot_surface(X2, Y2, Z2, alpha=0.25, color='lightblue')

    # ---------- 3. 绘制法向量 ---------- #
    ax.quiver(*center1, *normal1, length=5, color='green', linewidth=2, arrow_length_ratio=0.3)
    ax.quiver(*center2, *normal2, length=5, color='blue',  linewidth=2, arrow_length_ratio=0.3)

    # ---------- 4. 轴与图例设置 ---------- #
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    ax.view_init(elev=25, azim=40)
    plt.show()

def visualize_plane_normals(planes, normals, vis):
    """
    绘制平面的法向量
    
    参数:
    planes: 平面点云列表
    normals: 法向量列表
    vis: Open3D可视化器
    """
    for i, (plane_points, normal) in enumerate(zip(planes, normals)):
        # 计算平面的中心点
        center = np.mean(plane_points, axis=0)
        
        # 创建法向量的起点和终点
        start_point = center
        end_point = center + normal * 50  # 法向量长度设为50
        
        # 创建箭头
        points = np.array([start_point, end_point])
        lines = np.array([[0, 1]])  # 连接起点和终点
        
        # 创建LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # 设置线条颜色
        colors = np.array([[1, 0, 0] for _ in range(len(lines))])  # 红色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # 添加到可视化器
        vis.add_geometry(line_set)
        
        # 添加箭头头部
        arrow_head = o3d.geometry.TriangleMesh.create_cone(radius=5, height=10)
        # 计算箭头头部的旋转
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.arccos(np.dot(z_axis, normal) / (np.linalg.norm(z_axis) * np.linalg.norm(normal)))
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            arrow_head.rotate(R, center=[0, 0, 0])
        
        # 移动箭头头部到终点
        arrow_head.translate(end_point)
        arrow_head.paint_uniform_color([1, 0, 0])  # 红色
        
        # 添加到可视化器
        vis.add_geometry(arrow_head)

def compute_tree_plane_normals(tree_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """根据树形数据计算分支平面与主干平面的法向量
    
    Args:
        tree_data: 树形 JSON 数据 (包含 branches 列表)
    Returns:
        (normal_branch, normal_trunk) : 两个单位法向量
    """
    trunk_pts, br1_pts, br2_pts = find_max_points_branches(tree_data)
    branch_pts = np.vstack([br1_pts, br2_pts])
    # 分支平面法向量
    normal_branch, _ = calculate_plane_equation(branch_pts)
    normal_branch = normal_branch / np.linalg.norm(normal_branch)
    # 主干平面法向量 (要求与 branch 平面垂直)
    normal_trunk, _ = calculate_perpendicular_plane(trunk_pts, normal_branch)
    normal_trunk = normal_trunk / np.linalg.norm(normal_trunk)
    return normal_branch, normal_trunk

def generate_noisy_normals(base_normal: np.ndarray, n: int, max_angle_deg: float = 20.0, seed: int = None) -> np.ndarray:
    """围绕基准法向量生成带有角度扰动的法向量
    
    Args:
        base_normal : 基准单位法向量 (3,)
        n           : 生成数量
        max_angle_deg : 在每个欧拉角方向的最大偏转角度(度)
        seed        : 随机种子 (可选)
    Returns:
        noisy_normals : (n,3) 的单位向量数组
    """
    if seed is not None:
        np.random.seed(seed)
    base_normal = base_normal / np.linalg.norm(base_normal)
    noisy_list = []
    for _ in range(n):
        # 在 (-max_angle, max_angle) 范围随机采样三个欧拉角
        angles = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg, size=3))
        rx, ry, rz = angles
        # 分别构建绕 x/y/z 轴的小角度旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(rx), -np.sin(rx)],
                        [0, np.sin(rx),  np.cos(rx)]])
        R_y = np.array([[ np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
        R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                        [np.sin(rz),  np.cos(rz), 0],
                        [0, 0, 1]])
        # 组合旋转 (Z * Y * X)
        R = R_z @ R_y @ R_x
        new_normal = R @ base_normal
        new_normal /= np.linalg.norm(new_normal)
        noisy_list.append(new_normal)
    return np.array(noisy_list)

def calculate_normal_vector(points):
    # 计算点云的中心点
    centroid = np.mean(points, axis=0)
    
    # 计算协方差矩阵
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points.T)
    
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 最小特征值对应的特征向量即为法向量
    normal = eigenvectors[:, 0]
    
    # 确保法向量指向外部
    vectors_to_center = points - centroid
    dot_products = np.dot(vectors_to_center, normal)
    if np.sum(dot_products < 0) > len(points) / 2:
        normal = -normal
    
    return normal / np.linalg.norm(normal)

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

    # 计算单个点云的法向量
    # normal = calculate_normal_vector(trunk_points)

    # 可视化点云及其法向量
    # visualize_with_normals(trunk_points, normal_scale=5.0)

    # # 可视化多个平面及其法向量
    # planes = [trunk_points, trunk_points]
    # normals = [normal, normal]
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # visualize_plane_normals(planes, normals, vis)
    # vis.run() 