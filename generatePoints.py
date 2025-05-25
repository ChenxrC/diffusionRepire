import numpy as np
import json
from typing import List, Tuple, Dict, Any
import open3d as o3d

def generate_branch(start_point: np.ndarray, direction: np.ndarray, length: float, num_points: int) -> np.ndarray:
    """生成一个分支的点云数据，带有随机弯曲效果
    
    Args:
        start_point: 起始点坐标
        direction: 分支方向向量
        length: 分支长度
        num_points: 分支上的点数
    
    Returns:
        分支上的点云数据
    """
    # 确保最小长度为10
    length = max(length, 5.0)
    
    # 计算终点
    end_point = start_point + direction * length
    
    # 使用线段生成函数生成基础点
    points = generate_line_between_points(start_point, end_point)
    
    # 添加随机弯曲效果
    bend_angle = np.random.uniform(-30, 30)  # 随机弯曲角度
    bend_direction = np.random.randn(3)  # 随机弯曲方向
    bend_direction = bend_direction / np.linalg.norm(bend_direction)
    
    # 对每个点添加随机扰动
    for i in range(len(points)):
        t = i / (len(points) - 1)
        # 计算弯曲效果
        bend_factor = np.sin(t * np.pi) * bend_angle
        rotation_matrix = np.array([
            [np.cos(np.radians(bend_factor)), -np.sin(np.radians(bend_factor)), 0],
            [np.sin(np.radians(bend_factor)), np.cos(np.radians(bend_factor)), 0],
            [0, 0, 1]
        ])
        # 添加随机扰动
        noise = np.random.normal(0, 0.05, 3)
        points[i] = points[i] + noise
    
    return points

def calculate_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """计算绕任意轴的旋转矩阵
    
    Args:
        axis: 旋转轴向量
        angle: 旋转角度（度）
    
    Returns:
        旋转矩阵
    """
    # 确保旋转轴是单位向量
    axis = axis / np.linalg.norm(axis)
    
    # 将角度转换为弧度
    angle_rad = np.radians(angle)
    
    # 计算旋转矩阵
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # 构建旋转矩阵
    R = np.array([
        [cos_theta + axis[0]**2 * (1 - cos_theta),
         axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta,
         axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta],
        [axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta,
         cos_theta + axis[1]**2 * (1 - cos_theta),
         axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta],
        [axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta,
         axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta,
         cos_theta + axis[2]**2 * (1 - cos_theta)]
    ])
    
    return R

def generate_tree(levels: int = 3, points_per_branch: int = 10) -> Dict[str, Any]:
    """生成树形点云数据
    
    Args:
        levels: 树的层数
        points_per_branch: 每个分支上的点数
    
    Returns:
        包含树结构和点云数据的字典
    """
    tree_data = {
        "type": "tree",
        "levels": levels,
        "points_per_branch": points_per_branch,
        "branches": []
    }
    
    # 主干
    trunk_direction = np.array([0, 0, 1])
    trunk_length = points_per_branch
    trunk_points = generate_branch(
        np.array([0, 0, 0]),
        trunk_direction,
        trunk_length,
        points_per_branch
    )
    
    trunk_branch = {
        "type": "trunk",
        "level": 0,
        "direction": trunk_direction.tolist(),
        "length": trunk_length,
        "points": trunk_points.tolist(),
        "children": []
    }
    tree_data["branches"].append(trunk_branch)
    
    # 递归生成分支
    def generate_branches(parent_branch: Dict[str, Any], direction: np.ndarray, length: float, level: int):
        if level >= levels:
            return
            
        # 获取父分支的最后两个点来计算实际方向
        parent_points = np.array(parent_branch["points"])
        if len(parent_points) >= 2:
            actual_direction = parent_points[-1] - parent_points[-2]
            actual_direction = actual_direction / np.linalg.norm(actual_direction)
        else:
            actual_direction = direction
            
        # 生成2-3个分支
        num_branches = np.random.randint(2, 3)
        angles = np.linspace(-30, 30, num_branches)
        
        # 计算垂直于实际方向的向量作为旋转轴
        if abs(np.dot(actual_direction, np.array([0, 0, 1]))) > 0.9:
            perpendicular = np.array([1, 0, 0])
        else:
            perpendicular = np.cross(actual_direction, np.array([0, 0, 1]))
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        for angle in angles:
            # 随机旋转
            random_rotation = np.random.uniform(-20, 20)
            total_angle = angle + random_rotation
            
            # 计算新分支的方向
            # 首先绕实际方向旋转
            rotation_matrix = calculate_rotation_matrix(perpendicular, total_angle)
            new_direction = rotation_matrix @ actual_direction
            print(f'new_direction {new_direction}, level {level}, rotation_matrix {rotation_matrix}, actual_direction {actual_direction}, angle {angle}, random_rotation {random_rotation}')
            
            # 增加分支长度
            length_factor = np.random.uniform(1, 1.3)
            new_length = max(length * length_factor, 15.0)
            
            # 生成分支点
            start_point = np.array(parent_branch["points"][-1])
            branch_points = generate_branch(
                start_point,
                new_direction,
                new_length,
                points_per_branch
            )
            
            # 创建分支数据
            branch_data = {
                "type": "branch",
                "level": level,
                "direction": new_direction.tolist(),
                "length": new_length,
                "points": branch_points.tolist(),
                "children": []
            }
            
            # 添加到父分支的子节点中
            parent_branch["children"].append(branch_data)
            
            # 递归生成下一层
            generate_branches(branch_data, new_direction, new_length, level + 1)
    
    # 从主干开始生成分支
    generate_branches(trunk_branch, trunk_direction, trunk_length, 1)
    
    return tree_data

def save_to_json(tree_data: Dict[str, Any], filename: str = "tree_pointcloud.json"):
    """将树形点云数据保存为JSON文件
    
    Args:
        tree_data: 树形点云数据
        filename: 输出文件名
    """
    with open(filename, 'w') as f:
        json.dump(tree_data, f, indent=2)

def visualize_tree(tree_data: Dict[str, Any], planes: List[np.ndarray] = None):
    """可视化树形点云和平面"""
    points = extract_all_points(tree_data)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 1, 0])

    vis = o3d.visualization.Visualizer()
    success = vis.create_window()
    if not success:
        print("Open3D窗口创建失败，可能是因为当前环境没有图形界面（如SSH无X11）。")
        print("请在有GUI的环境下运行，或使用X11转发（ssh -X）等方式。")
        return

    vis.add_geometry(pcd)
    if planes:
        for plane_points in planes:
            plane_pcd = o3d.geometry.PointCloud()
            plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
            plane_pcd.paint_uniform_color([1, 0, 0])
            vis.add_geometry(plane_pcd)

    view_ctl = vis.get_view_control()
    if view_ctl is not None:
        view_ctl.set_front([0, 0, -1])
        view_ctl.set_lookat([0, 0, 0])
        view_ctl.set_up([0, 1, 0])

    vis.run()
    vis.destroy_window()

def generate_line_between_points(start_point: np.ndarray, end_point: np.ndarray) -> np.ndarray:
    """在两点之间生成等间距的点
    
    Args:
        start_point: 起始点坐标
        end_point: 终点坐标
    
    Returns:
        两点之间的等间距点云数据
    """
    # 计算方向向量
    direction = end_point - start_point
    total_length = np.linalg.norm(direction)
    direction = direction / total_length  # 单位化方向向量
    
    # 计算需要生成的点数（向上取整）
    num_points = int(np.ceil(total_length)) + 1
    
    # 生成等间距点
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = start_point + direction * total_length * t
        points.append(point.tolist())
    
    return np.array(points)

if __name__ == "__main__":
    # 生成树形点云数据
    tree_data = generate_tree(levels=3, points_per_branch=15)
    
    # 保存为JSON文件
    save_to_json(tree_data)
    
    # 计算总点数
    def count_points(branch):
        points = len(branch["points"])
        for child in branch["children"]:
            points += count_points(child)
        return points
    
    total_points = sum(count_points(branch) for branch in tree_data["branches"])
    print(f"生成了 {total_points} 个点")
    print(f"数据已保存到 tree_pointcloud.json")

    # 批量生成多棵树
    forest = [generate_tree(levels=3, points_per_branch=10) for _ in range(5)]
    with open('forest_pointcloud.json', 'w') as f:
        json.dump(forest, f, indent=2) 