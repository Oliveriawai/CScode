import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 读取图像
    img = cv2.imread('fruits.jpg')
    
    if img is None:
        print("错误：找不到 fruits.jpg")
        print("请将图像文件放在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    # 2. 转换为RGB用于显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. 交互式选择种子点
    print("\n请在弹出的图像窗口中点击水果区域，按ESC退出")
    
    # 创建窗口并设置鼠标回调
    seed_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            seed_points.append((x, y))
            print(f"选择种子点：({x}, {y})")
            # 在图像上标记种子点
            img_copy = img_rgb.copy()
            cv2.circle(img_copy, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow('Select Seed Points - Click on fruits', img_copy)
    
    cv2.imshow('Select Seed Points - Click on fruits', img_rgb)
    cv2.setMouseCallback('Select Seed Points - Click on fruits', mouse_callback)
    
    print("\n操作说明：")
    print("- 鼠标左键点击选择种子点")
    print("- 按 'q' 完成选择并开始分割")
    print("- 按 'r' 清除所有种子点")
    print("- 按 ESC 取消")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            seed_points = []
            print("已清除所有种子点")
            cv2.imshow('Select Seed Points - Click on fruits', img_rgb)
        elif key == 27:
            cv2.destroyAllWindows()
            return
    
    cv2.destroyAllWindows()
    
    if len(seed_points) == 0:
        # 如果没有选择种子点，使用默认种子点
        seed_points = [(200, 150), (400, 200), (300, 350)]
        print(f"\n使用默认种子点：{seed_points}")
    
    # 4. 漫水填充分割
    # 创建掩码（比原图大2像素）
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # 设置参数
    lo_diff = (20, 20, 20)  # 向下差异阈值
    up_diff = (20, 20, 20)  # 向上差异阈值
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE  # 4连通，固定范围
    
    # 存储分割结果
    results = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128)]
    
    img_result = img_rgb.copy()
    img_mask_vis = img_rgb.copy()
    
    for i, (x, y) in enumerate(seed_points):
        if y >= h or x >= w:
            print(f"种子点({x},{y})超出图像范围，跳过")
            continue
        
        # 复制掩码（每个种子点使用新的掩码）
        mask_i = mask.copy()
        
        # 漫水填充
        color = colors[i % len(colors)]
        ret, img_result, mask_i, rect = cv2.floodFill(img_result, mask_i, (x, y), color, 
                                                        lo_diff, up_diff, flags)
        
        print(f"种子点{i+1}：填充了{ret}个像素，颜色{color}")
        
        # 可视化掩码区域
        mask_region = mask_i[1:h+1, 1:w+1]
        img_mask_vis[mask_region == 1] = color
    
    # 5. 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    # 标记种子点
    img_with_seeds = img_rgb.copy()
    for x, y in seed_points:
        cv2.circle(img_with_seeds, (x, y), 5, (255, 0, 0), -1)
    plt.imshow(img_with_seeds)
    plt.title(f'Seed Points ({len(seed_points)} points)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_result)
    plt.title('Flood Fill Segmentation Result')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(img_mask_vis)
    plt.title('Mask Visualization (Different Colors)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_5.2_result.png', dpi=150)
    plt.show()

def compare_parameters():
    """比较不同参数对漫水填充效果的影响"""
    img = cv2.imread('fruits.jpg')
    
    if img is None:
        print("错误：找不到 fruits.jpg")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 固定种子点（图像中心偏左上的一个水果）
    seed_point = (200, 150)
    
    # 不同参数组合
    param_sets = [
        (5, 5, 4, "Low threshold (5,5), 4-connectivity"),
        (20, 20, 4, "Medium threshold (20,20), 4-connectivity"),
        (50, 50, 4, "High threshold (50,50), 4-connectivity"),
        (20, 20, 8, "Medium threshold (20,20), 8-connectivity"),
        (20, 20, 4 | cv2.FLOODFILL_FIXED_RANGE, "Fixed range mode"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    cv2.circle(img_rgb, seed_point, 5, (255, 0, 0), -1)
    
    for i, (lo, up, flag, title) in enumerate(param_sets[:5]):
        img_result = img_rgb.copy()
        mask = np.zeros((h+2, w+2), np.uint8)
        color = (0, 255, 0)
        
        ret, img_result, mask, rect = cv2.floodFill(img_result, mask, seed_point, 
                                                     color, (lo, lo, lo), (up, up, up), flag)
        
        row = (i + 1) // 3
        col = (i + 1) % 3
        axes[row, col].imshow(img_result)
        axes[row, col].set_title(f'ret={ret}\n{title[:30]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_5.2_compare.png', dpi=150)
    plt.show()
    
    print("\n参数影响分析：")
    print("阈值越大，填充范围越大，可能溢出到相邻区域")
    print("8连通比4连通填充更充分，但可能连接不相关的区域")
    print("固定范围模式基于像素值差异，浮动范围模式考虑颜色变化")

def segment_specific_fruit():
    """分割特定水果（通过选择种子点）"""
    img = cv2.imread('fruits.jpg')
    
    if img is None:
        print("错误：找不到 fruits.jpg")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    print("\n请选择要分割的水果（在图像上点击）：")
    
    seed_point = None
    
    def select_seed(event, x, y, flags, param):
        nonlocal seed_point
        if event == cv2.EVENT_LBUTTONDOWN:
            seed_point = (x, y)
            print(f"选择种子点：({x}, {y})")
            img_display = img_rgb.copy()
            cv2.circle(img_display, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow('Select Fruit - Click on the fruit', img_display)
    
    cv2.imshow('Select Fruit - Click on the fruit', img_rgb)
    cv2.setMouseCallback('Select Fruit - Click on the fruit', select_seed)
    
    while seed_point is None:
        if cv2.waitKey(100) & 0xFF == 27:
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    
    # 调整阈值
    thresholds = [(10, 10), (20, 20), (30, 30), (40, 40)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    cv2.circle(img_rgb, seed_point, 5, (255, 0, 0), -1)
    
    for i, (lo, up) in enumerate(thresholds):
        img_result = img_rgb.copy()
        mask = np.zeros((h+2, w+2), np.uint8)
        
        ret, img_result, mask, rect = cv2.floodFill(img_result, mask, seed_point, 
                                                     (0, 255, 0), (lo, lo, lo), (up, up, up))
        
        row = (i + 1) // 3
        col = (i + 1) % 3
        axes[row, col].imshow(img_result)
        axes[row, col].set_title(f'Threshold: {lo}-{up}\nPixels: {ret}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_5.2_fruit_seg.png', dpi=150)
    plt.show()

def interactive_floodfill():
    """交互式漫水填充（实时调整阈值）"""
    img = cv2.imread('fruits.jpg')
    
    if img is None:
        print("错误：找不到 fruits.jpg")
        return
    
    img_display = img.copy()
    h, w = img.shape[:2]
    
    # 全局变量
    lo_diff = 20
    up_diff = 20
    seed_point = None
    
    def update_floodfill():
        nonlocal img_display
        if seed_point is None:
            return
        
        img_display = img.copy()
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # 随机颜色
        color = (np.random.randint(100, 255), 
                 np.random.randint(100, 255), 
                 np.random.randint(100, 255))
        
        ret, img_display, mask, rect = cv2.floodFill(img_display, mask, seed_point, 
                                                      color, (lo_diff, lo_diff, lo_diff), 
                                                      (up_diff, up_diff, up_diff))
        
        # 显示信息
        cv2.putText(img_display, f'Lo: {lo_diff} Up: {up_diff} Pixels: {ret}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Flood Fill Interactive', img_display)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal seed_point
        if event == cv2.EVENT_LBUTTONDOWN:
            seed_point = (x, y)
            update_floodfill()
    
    def on_lo_change(val):
        nonlocal lo_diff
        lo_diff = val
        update_floodfill()
    
    def on_up_change(val):
        nonlocal up_diff
        up_diff = val
        update_floodfill()
    
    cv2.namedWindow('Flood Fill Interactive')
    cv2.setMouseCallback('Flood Fill Interactive', mouse_callback)
    cv2.createTrackbar('Low Diff', 'Flood Fill Interactive', lo_diff, 100, on_lo_change)
    cv2.createTrackbar('Up Diff', 'Flood Fill Interactive', up_diff, 100, on_up_change)
    
    cv2.imshow('Flood Fill Interactive', img)
    
    print("\n交互模式说明：")
    print("- 鼠标左键点击选择种子点")
    print("- 调整滑动条改变阈值范围")
    print("- 按 ESC 退出")
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()

def segment_multiple_regions():
    """分割多个区域并统计"""
    img = cv2.imread('fruits.jpg')
    
    if img is None:
        print("错误：找不到 fruits.jpg")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 预定义多个种子点（不同水果）
    seed_points = [
        (150, 100, "红色苹果"),
        (350, 120, "橙色橘子"),
        (250, 300, "绿色苹果"),
        (450, 350, "香蕉"),
        (100, 380, "紫色葡萄"),
    ]
    
    img_result = img_rgb.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    results = []
    
    for i, (x, y, name) in enumerate(seed_points):
        if y >= h or x >= w:
            continue
        
        mask_i = mask.copy()
        ret, img_result, mask_i, rect = cv2.floodFill(img_result, mask_i, (x, y), 
                                                        colors[i % len(colors)],
                                                        (25, 25, 25), (25, 25, 25))
        
        results.append((name, ret))
        print(f"{name}：填充了{ret}个像素")
    
    # 显示结果
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_result)
    plt.title('Multi-region Segmentation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_5.2_multi.png', dpi=150)
    plt.show()
    
    # 输出统计
    print("\n各区域面积统计：")
    total = 0
    for name, area in results:
        print(f"  {name}：{area} 像素")
        total += area
    
    # 计算面积占比
    img_area = h * w
    print(f"\n总填充面积：{total} / {img_area} ({100*total/img_area:.1f}%)")

if __name__ == "__main__":
    print("实验5.2：浸水算法分割（Flood Fill）")
    print("="*50)
    print("1. 基础实验（选择种子点分割）")
    print("2. 比较不同参数效果")
    print("3. 分割特定水果（选择种子点）")
    print("4. 交互式漫水填充（实时调整阈值）")
    print("5. 分割多个区域并统计")
    
    choice = input("请选择 (1/2/3/4/5): ")
    
    if choice == "2":
        compare_parameters()
    elif choice == "3":
        segment_specific_fruit()
    elif choice == "4":
        interactive_floodfill()
    elif choice == "5":
        segment_multiple_regions()
    else:
        main()