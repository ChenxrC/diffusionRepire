# 血管曲面生成验证工具使用指南

本指南介绍如何使用血管曲面生成的验证和可视化工具来校验扩散模型生成的曲面质量。

## 文件结构

```
diffusionRepire/
├── tree_plane_diffusion.py      # 主要训练和生成代码（包含验证函数）
├── validate_surface.py          # 独立验证脚本
├── quick_validation_example.py  # 快速验证演示
└── README_validation.md         # 本文档
```

## 核心验证功能

### 1. 基础可视化函数

#### `visualize_generated_surface(tree_json, generated_points, save_path, show_wireframe)`
生成三联图对比显示：
- **左图**: 原始血管树（主干、分支1、分支2）
- **中图**: 生成的曲面（带网格线）
- **右图**: 叠加显示（血管点 + 半透明曲面）

```python
from tree_plane_diffusion import visualize_generated_surface

# 可视化生成的曲面
visualize_generated_surface(
    tree_json="tree_1.json",
    generated_points=surface_array,  # shape: (32, 32, 3)
    save_path="surface_visual.png",
    show_wireframe=True
)
```

### 2. 质量分析函数

#### `analyze_surface_quality(tree_json, generated_points)`
返回详细的质量指标字典：

**覆盖范围分析**:
- `vessel_bbox`: 血管边界框
- `surface_bbox`: 曲面边界框  
- `coverage_ratio`: 覆盖比例

**中心对齐分析**:
- `center_distance`: 曲面中心与血管中心的距离

**拟合精度分析**:
- `min_distance_stats`: 血管点到曲面最小距离的统计信息
  - `mean`: 平均距离
  - `std`: 标准差
  - `median`: 中位数
  - `max/min`: 最大/最小距离

**平滑度分析**:
- `smoothness`: 水平和垂直方向的相邻点距离统计

**法向量一致性**:
- `normal_consistency`: 相邻法向量角度差的统计

### 3. 综合验证函数

#### `comprehensive_surface_validation(tree_json, generated_points, save_prefix)`
一站式验证，包括：
- 自动生成可视化图像
- 执行质量分析
- 打印详细报告
- 保存JSON格式的指标文件

```python
from tree_plane_diffusion import comprehensive_surface_validation

# 综合验证
metrics = comprehensive_surface_validation(
    tree_json="tree_1.json",
    generated_points=surface_array,
    save_prefix="validation_result"
)

# 输出文件:
# - validation_result_visualization.png
# - validation_result_metrics.json
```

## 使用方式

### 方式1: 直接在代码中调用

```python
import numpy as np
from tree_plane_diffusion import (
    train_tree_diffusion,
    denoise_with_tree,
    comprehensive_surface_validation
)
from validate_surface import (
    generate_ideal_surface_from_tree,
    validate_ideal_surface_only,
    compare_ideal_vs_generated
)

# 1. 训练模型
model, betas = train_tree_diffusion(['tree_1.json'], epochs=5000)

# 2. 生成曲面
surface = denoise_with_tree('tree_1.json', model, betas)

# 3. 验证
comprehensive_surface_validation('tree_1.json', surface, "my_validation")

# 4. 新增：理想曲面验证
ideal_surface = validate_ideal_surface_only('tree_1.json')

# 5. 新增：理想曲面与生成曲面对比
ideal_surf, metrics = compare_ideal_vs_generated('tree_1.json', surface)
```

### 方式2: 使用独立验证脚本

```bash
# 单个曲面验证
python validate_surface.py --tree_json tree_1.json --model_path model.pth --mode single

# 多样本比较（生成5个样本对比）
python validate_surface.py --tree_json tree_1.json --model_path model.pth --mode multiple --n_samples 5

# 详细分析（包括几何特性）
python validate_surface.py --tree_json tree_1.json --model_path model.pth --mode detailed

# 泛化能力测试（用所有血管树测试）
python validate_surface.py --tree_json tree_1.json --model_path model.pth --mode generalization

# 新增：理想曲面验证（不需要模型）
python validate_surface.py --tree_json tree_1.json --mode ideal_surface

# 新增：理想曲面与生成曲面对比
python validate_surface.py --tree_json tree_1.json --model_path model.pth --mode ideal_comparison
```

### 方式3: 快速演示

```bash
# 运行交互式演示
python quick_validation_example.py

# 选择菜单选项：
# 1. 快速验证演示
# 2. 交互式验证
# 3. 批量验证演示
# 4. 理想曲面演示
# 5. 理想曲面 vs 生成曲面对比
# 6. 批量理想曲面生成

# 或者直接运行特定演示
python -c "from quick_validation_example import ideal_surface_demo; ideal_surface_demo()"
```

## 新增：理想曲面验证功能

### 1. 理想曲面生成原理

理想曲面基于以下数据生成：
- **血管主干点**: 提供曲面的主要走向
- **分支对应点的中点**: 两个分支在相同位置的点的中点，提供曲面的横向范围
- **PCA主成分分析**: 确定曲面的主平面方向
- **距离相关弯曲**: 添加轻微的自然弯曲效果

### 2. 理想曲面验证模式

#### `ideal_surface` 模式
只生成和验证理想曲面，不需要训练模型：

```bash
python validate_surface.py --tree_json tree_1.json --mode ideal_surface --grid_size 32
```

**输出文件**:
- `ideal_surface_visualization.png`: 理想曲面三联图
- `ideal_surface_metrics.json`: 理想曲面质量指标

#### `ideal_comparison` 模式
对比理想曲面与扩散模型生成曲面：

```bash
python validate_surface.py --tree_json tree_1.json --model_path model.pth --mode ideal_comparison
```

**输出文件**:
- `comparison_ideal_vs_generated.png`: 四联图对比
  - 理想曲面
  - 生成曲面
  - 点对点差异热图
  - 叠加对比
- `comparison_ideal_comparison_metrics.json`: 差异指标

### 3. 理想曲面对比指标

对比分析提供以下差异指标：

```json
{
  "mean_point_distance": 2.34,      // 平均点距差异
  "std_point_distance": 1.12,       // 差异标准差
  "max_point_distance": 8.67,       // 最大点距差异
  "median_point_distance": 1.89,    // 差异中位数
  "rmse": 2.58                      // 均方根误差
}
```

### 4. 理想曲面质量标准

- **平均点距差异 < 3.0**: 生成曲面与理想曲面非常接近
- **平均点距差异 3.0-6.0**: 接近程度一般，有改进空间
- **平均点距差异 > 6.0**: 差异较大，需要调整模型

### 5. 使用场景

#### 快速质量评估
```python
from validate_surface import validate_ideal_surface_only

# 快速生成理想曲面作为参考
ideal_surface = validate_ideal_surface_only('tree_1.json')
```

#### 模型性能基准
```python
from validate_surface import compare_ideal_vs_generated

# 将生成曲面与理想曲面对比
ideal_surf, metrics = compare_ideal_vs_generated('tree_1.json', generated_surface)
print(f"与理想曲面的RMSE: {metrics['rmse']:.4f}")
```

#### 批量基准测试
```python
# 为多个血管树生成理想曲面基准
tree_files = glob.glob('tree_*.json')
for tree_file in tree_files:
    ideal_surface = generate_ideal_surface_from_tree(tree_file)
    # 保存作为该血管树的基准曲面
```

## 验证指标解读

### 1. 覆盖范围分析
- **覆盖比例 > 0.8**: 曲面很好地覆盖了血管区域
- **覆盖比例 0.5-0.8**: 覆盖适中，可能有部分区域未覆盖
- **覆盖比例 < 0.5**: 覆盖不足，需要调整模型

### 2. 中心对齐分析
- **中心距离 < 5.0**: 曲面中心与血管中心对齐良好
- **中心距离 5.0-15.0**: 对齐一般，可能有偏移
- **中心距离 > 15.0**: 对齐较差，需要检查

### 3. 拟合精度分析
- **平均距离 < 2.0**: 拟合精度很好
- **平均距离 2.0-5.0**: 拟合精度一般
- **平均距离 > 5.0**: 拟合精度较差

### 4. 平滑度分析
- **标准差 < 1.0**: 曲面很平滑
- **标准差 1.0-3.0**: 平滑度一般
- **标准差 > 3.0**: 曲面可能有明显的突变

## 高级验证功能

### 多样本一致性检验
```python
from validate_surface import compare_multiple_generations

# 生成多个样本比较差异
samples = compare_multiple_generations(
    tree_json="tree_1.json",
    model=model,
    betas=betas,
    n_samples=5
)
# 自动计算样本间差异并可视化
```

### 详细几何分析
```python
from validate_surface import detailed_surface_analysis

# 分析曲率、表面积、厚度等几何特性
detailed_surface_analysis("tree_1.json", surface_array)
```

### 泛化能力测试
```python
from validate_surface import validate_with_different_trees

# 用多个不同血管树测试模型泛化能力
results = validate_with_different_trees(
    model_path="model.pth",
    tree_files=["tree_1.json", "tree_2.json", "tree_3.json"]
)
```

## 常见问题处理

### 1. 导入错误
确保所有依赖文件在同一目录：
```python
# 检查必要文件
import os
required_files = ['tree_plane_diffusion.py', 'visual.py', 'tree_plane_predictor.py']
for f in required_files:
    if not os.path.exists(f):
        print(f"缺少文件: {f}")
```

### 2. 可视化不显示
如果使用SSH或无图形界面环境：
```python
# 确保保存而不是显示
visualize_generated_surface(
    tree_json="tree_1.json",
    generated_points=surface,
    save_path="output.png"  # 指定保存路径
)
```

### 3. 内存不足
对于大网格，可以降低分辨率：
```python
# 使用较小的网格
surface = denoise_with_tree(tree_json, model, betas, grid_size=16)  # 默认32
```

## 输出文件说明

### 可视化图像 (.png)
- **三联图**: 原始血管 | 生成曲面 | 叠加对比
- **网格线**: 显示曲面的结构化网格
- **颜色编码**: 不同分支用不同颜色区分

### 质量指标 (.json)
```json
{
  "vessel_bbox": {...},
  "surface_bbox": {...},
  "coverage_ratio": [...],
  "center_distance": 3.14,
  "min_distance_stats": {
    "mean": 1.23,
    "std": 0.45,
    ...
  },
  "smoothness": {...},
  "normal_consistency": {...}
}
```

### GIF动画 (.gif)
去噪过程的动态可视化，显示：
- 随机噪声点云逐步变为结构化曲面
- 每一步的去噪效果
- 最终收敛到目标曲面

## 最佳实践

1. **训练后立即验证**: 每次训练完成后运行综合验证
2. **多样本检验**: 生成多个样本检查一致性
3. **参数调优**: 根据验证结果调整训练参数
4. **可视化检查**: 始终进行可视化检查，数值指标可能无法反映所有问题
5. **保存记录**: 保存验证结果用于模型比较和改进

通过这套完整的验证体系，您可以全面评估扩散模型生成的血管曲面质量，确保生成结果符合预期。 