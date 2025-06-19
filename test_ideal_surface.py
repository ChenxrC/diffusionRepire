"""
测试理想曲面生成功能
"""

import os
import sys
import glob
import numpy as np

def test_ideal_surface():
    """测试理想曲面生成功能"""
    print("测试理想曲面生成功能...")
    
    # 查找血管树文件
    tree_files = glob.glob('tree_*.json')
    if not tree_files:
        print("错误：未找到血管树文件 (tree_*.json)")
        print("请确保当前目录下有血管树JSON文件")
        return False
    
    print(f"找到 {len(tree_files)} 个血管树文件: {tree_files}")
    
    try:
        from validate_surface import generate_ideal_surface_from_tree, validate_ideal_surface_only
        
        # 测试1: 基础理想曲面生成
        print("\n测试1: 基础理想曲面生成...")
        test_file = tree_files[0]
        print(f"使用文件: {test_file}")
        
        ideal_surface = generate_ideal_surface_from_tree(test_file, grid_size=16)
        print(f"✓ 成功生成理想曲面，形状: {ideal_surface.shape}")
        
        # 测试2: 完整验证流程
        print("\n测试2: 完整验证流程...")
        result = validate_ideal_surface_only(test_file, grid_size=16, save_prefix="test_ideal")
        print(f"✓ 完整验证流程成功")
        
        # 测试3: 批量处理
        print("\n测试3: 批量处理...")
        for i, tree_file in enumerate(tree_files[:3]):  # 只测试前3个文件
            print(f"处理文件 {i+1}: {tree_file}")
            surface = generate_ideal_surface_from_tree(tree_file, grid_size=12)
            print(f"  ✓ 生成曲面形状: {surface.shape}")
        
        print("\n🎉 所有测试通过！理想曲面生成功能正常工作。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_line():
    """测试命令行功能"""
    print("\n测试命令行功能...")
    
    tree_files = glob.glob('tree_*.json')
    if not tree_files:
        print("跳过命令行测试：没有血管树文件")
        return
    
    test_file = tree_files[0]
    
    # 测试理想曲面模式
    print("测试理想曲面模式...")
    cmd = f"python validate_surface.py --tree_json {test_file} --mode ideal_surface --grid_size 16 --output_prefix test_cmd"
    print(f"执行命令: {cmd}")
    
    import subprocess
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ 命令行理想曲面模式测试成功")
        else:
            print(f"❌ 命令行测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ 命令行测试超时")
    except Exception as e:
        print(f"❌ 命令行测试出错: {e}")

def main():
    """主测试函数"""
    print("=" * 60)
    print("理想曲面生成功能测试")
    print("=" * 60)
    
    # 检查当前目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 基础功能测试
    success = test_ideal_surface()
    
    if success:
        # 命令行测试
        test_command_line()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("现在您可以使用以下命令:")
        print("1. python validate_surface.py --tree_json tree_1.json --mode ideal_surface")
        print("2. python quick_validation_example.py  # 选择选项4或5")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("测试失败，请检查错误信息")
        print("=" * 60)

if __name__ == "__main__":
    main() 