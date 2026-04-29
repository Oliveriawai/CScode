"""
diagnose.py - 诊断脚本，检查环境和文件
"""
import os
import cv2
import numpy as np

print("=" * 60)
print("环境诊断")
print("=" * 60)

# 1. 检查OpenCV版本
print(f"\n1. OpenCV版本: {cv2.__version__}")

# 2. 检查当前工作目录
print(f"\n2. 当前工作目录: {os.getcwd()}")

# 3. 检查文件夹和文件
print("\n3. 检查必要文件:")
print("-" * 40)

# 检查 calib_images 文件夹
if os.path.exists("calib_images"):
    print("✓ calib_images 文件夹存在")
    images = [f for f in os.listdir("calib_images") if f.endswith('.bmp')]
    print(f"  找到 {len(images)} 个bmp文件: {sorted(images)}")
    
    # 检查第一张图片是否能读取
    if images:
        first_img = os.path.join("calib_images", sorted(images)[0])
        img = cv2.imread(first_img)
        if img is not None:
            print(f"  ✓ 可以成功读取图片: {sorted(images)[0]}, 尺寸: {img.shape}")
        else:
            print(f"  ✗ 无法读取图片: {first_img}")
else:
    print("✗ calib_images 文件夹不存在！")
    print("  请确保在正确目录下创建 calib_images 文件夹")

# 检查 robot_poses.txt
if os.path.exists("robot_poses.txt"):
    print("✓ robot_poses.txt 存在")
    with open("robot_poses.txt", 'r') as f:
        content = f.read()
        print(f"  文件大小: {len(content)} 字符")
        print(f"  前200字符:\n{content[:200]}")
else:
    print("✗ robot_poses.txt 不存在！")

# 4. 尝试简单的角点检测测试
print("\n4. 角点检测测试:")
print("-" * 40)

if os.path.exists("calib_images"):
    test_img_path = os.path.join("calib_images", "1.bmp")
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 尝试检测角点
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            print(f"  ✓ 成功检测到棋盘格角点！")
        else:
            print(f"  ✗ 未检测到棋盘格角点")
            print("    请确认棋盘格是9x6个内角点")
    else:
        print(f"  ✗ 找不到测试图片: {test_img_path}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)