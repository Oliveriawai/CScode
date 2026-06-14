import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 读取图像
    img = cv2.imread('coins.jpg')
    
    if img is None:
        print("错误：找不到 coins.jpg")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 预处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # 3. 大津阈值
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. 形态学操作（去除小噪声，但不破坏粘连区域）
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. 距离变换（关键：调整阈值来平衡粘连和过分割）
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    
    # 动态调整距离变换阈值（根据图像内容自适应）
    dist_max = dist_transform.max()
    
    # 尝试多个阈值，选择最优结果
    best_result = None
    best_count = 0
    
    # 阈值从0.3到0.7，步长0.05
    for ratio in np.arange(0.3, 0.75, 0.05):
        _, sure_fg = cv2.threshold(dist_transform, ratio * dist_max, 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 标记连通区域
        num_labels, labels = cv2.connectedComponents(sure_fg)
        
        # 计算候选硬币数量（排除背景）
        candidate_count = num_labels - 1
        
        # 选择中等数量的结果（不过多也不过少）
        if 15 <= candidate_count <= 25:
            best_result = sure_fg
            best_count = candidate_count
            best_ratio = ratio
            break
    
    if best_result is None:
        # 如果没有找到合适的结果，使用默认阈值0.5
        _, best_result = cv2.threshold(dist_transform, 0.5 * dist_max, 255, 0)
        best_result = np.uint8(best_result)
        best_ratio = 0.5
        # 标记连通区域获取数量
        num_labels, labels = cv2.connectedComponents(best_result)
        best_count = num_labels - 1
    
    # 6. 对结果进行后处理
    # 去除太小的区域（噪声）
    kernel_small = np.ones((2, 2), np.uint8)
    best_result = cv2.morphologyEx(best_result, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 重新标记连通区域
    num_labels, labels = cv2.connectedComponents(best_result)
    
    # 7. 计算每个硬币的半径
    radii = []
    centers = []
    areas = []
    
    for i in range(1, num_labels):
        # 获取第i个硬币的像素坐标
        coords = np.where(labels == i)
        if len(coords[0]) < 30:  # 忽略太小的区域
            continue
        
        # 计算质心
        cy = int(np.mean(coords[0]))
        cx = int(np.mean(coords[1]))
        centers.append((cx, cy))
        
        # 计算等效半径（从面积推算，更稳定）
        area = len(coords[0])
        radius = np.sqrt(area / np.pi)
        radii.append(radius)
        areas.append(area)
    
    # 8. 处理粘连硬币的合并（如果相邻太近，合并它们）
    merged = True
    while merged and len(centers) > 1:
        merged = False
        distances = []
        # 计算所有硬币中心之间的距离
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                              (centers[i][1] - centers[j][1])**2)
                distances.append((dist, i, j))
        
        # 按距离排序
        distances.sort()
        
        # 合并距离过近的硬币（距离小于半径之和）
        for dist, i, j in distances:
            if i >= len(centers) or j >= len(centers):
                continue
            if dist < (radii[i] + radii[j]) * 0.7:
                # 合并：取中心点，面积相加
                new_cx = (centers[i][0] * areas[i] + centers[j][0] * areas[j]) // (areas[i] + areas[j])
                new_cy = (centers[i][1] * areas[i] + centers[j][1] * areas[j]) // (areas[i] + areas[j])
                new_area = areas[i] + areas[j]
                new_radius = np.sqrt(new_area / np.pi)
                
                # 替换
                centers[i] = (new_cx, new_cy)
                radii[i] = new_radius
                areas[i] = new_area
                
                # 删除第j个
                centers.pop(j)
                radii.pop(j)
                areas.pop(j)
                merged = True
                break
    
    # 9. 统计结果
    num_coins = len(radii)
    avg_radius = np.mean(radii)
    var_radius = np.var(radii)
    std_radius = np.std(radii)
    
    print("\n" + "="*50)
    print("硬币检测统计结果")
    print("="*50)
    print(f"检测到的硬币数量：{num_coins}")
    print(f"平均半径：{avg_radius:.2f} 像素")
    print(f"半径方差：{var_radius:.2f}")
    print(f"半径标准差：{std_radius:.2f}")
    print(f"最大半径：{np.max(radii):.2f} 像素")
    print(f"最小半径：{np.min(radii):.2f} 像素")
    
    # 10. 绘制结果
    plt.figure(figsize=(16, 12))
    
    # 原图
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # 阈值分割结果
    plt.subplot(2, 3, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title('Threshold Segmentation')
    plt.axis('off')
    
    # 距离变换结果
    plt.subplot(2, 3, 3)
    plt.imshow(dist_transform, cmap='jet')
    plt.title(f'Distance Transform (threshold={best_ratio:.2f})')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # 标记硬币
    img_marked = img_rgb.copy()
    for i, (cx, cy) in enumerate(centers):
        radius = radii[i]
        cv2.circle(img_marked, (cx, cy), int(radius), (0, 255, 0), 2)
        cv2.putText(img_marked, f'{i+1}', (cx-10, cy-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    plt.subplot(2, 3, 4)
    plt.imshow(img_marked)
    plt.title(f'Detected Coins ({num_coins} coins)')
    plt.axis('off')
    
    # 半径分布直方图
    plt.subplot(2, 3, 5)
    plt.hist(radii, bins=15, color='gold', alpha=0.7, edgecolor='black')
    plt.axvline(avg_radius, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_radius:.1f}')
    plt.title('Radius Distribution')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 面积分布
    plt.subplot(2, 3, 6)
    plt.bar(range(1, num_coins+1), areas, color='steelblue', alpha=0.7)
    plt.axhline(np.mean(areas), color='red', linestyle='--', label=f'Mean: {np.mean(areas):.0f}')
    plt.title('Coin Area Distribution')
    plt.xlabel('Coin Index')
    plt.ylabel('Area (pixels²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hw_7_1_result.png', dpi=150)
    plt.show()
    
    return num_coins, avg_radius, std_radius

def watershed_with_adaptive_markers():
    """改进的分水岭算法（避免过分割）"""
    img = cv2.imread('coins.jpg')
    
    if img is None:
        print("错误：找不到 coins.jpg")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 预处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # 2. 大津阈值
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. 距离变换
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist_max = dist_transform.max()
    
    # 5. 自适应确定前景区域（避免过分割）
    # 使用较高的阈值来确保每个硬币只有一个种子点
    sure_fg = np.uint8(dist_transform > 0.6 * dist_max)
    
    # 6. 确定背景区域
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    
    # 7. 标记未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 8. 连通组件标记（添加边界避免过分割）
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 背景标记为1
    markers[unknown == 255] = 0  # 未知区域标记为0
    
    # 9. 应用分水岭
    markers = cv2.watershed(img, markers)
    
    # 10. 后处理：合并过度分割的区域
    img_marked = img_rgb.copy()
    radii = []
    centers = []
    
    # 收集每个区域的面积和中心
    region_data = {}
    for label in range(2, np.max(markers) + 1):
        coords = np.where(markers == label)
        if len(coords[0]) < 100:  # 忽略太小的区域
            continue
        
        area = len(coords[0])
        cy = int(np.mean(coords[0]))
        cx = int(np.mean(coords[1]))
        
        region_data[label] = {
            'coords': coords,
            'area': area,
            'center': (cx, cy),
            'radius': np.sqrt(area / np.pi)
        }
    
    # 合并距离过近的区域
    labels_list = list(region_data.keys())
    merged = True
    while merged and len(labels_list) > 1:
        merged = False
        for i in range(len(labels_list)):
            for j in range(i+1, len(labels_list)):
                label_i = labels_list[i]
                label_j = labels_list[j]
                ci = region_data[label_i]['center']
                cj = region_data[label_j]['center']
                ri = region_data[label_i]['radius']
                rj = region_data[label_j]['radius']
                dist = np.sqrt((ci[0]-cj[0])**2 + (ci[1]-cj[1])**2)
                
                if dist < (ri + rj) * 0.6:
                    # 合并
                    merged_area = region_data[label_i]['area'] + region_data[label_j]['area']
                    merged_cx = (ci[0]*region_data[label_i]['area'] + cj[0]*region_data[label_j]['area']) // merged_area
                    merged_cy = (ci[1]*region_data[label_i]['area'] + cj[1]*region_data[label_j]['area']) // merged_area
                    
                    # 合并坐标
                    merged_coords = (np.concatenate([region_data[label_i]['coords'][0], region_data[label_j]['coords'][0]]),
                                     np.concatenate([region_data[label_i]['coords'][1], region_data[label_j]['coords'][1]]))
                    
                    region_data[label_i] = {
                        'coords': merged_coords,
                        'area': merged_area,
                        'center': (merged_cx, merged_cy),
                        'radius': np.sqrt(merged_area / np.pi)
                    }
                    del region_data[label_j]
                    labels_list.remove(label_j)
                    merged = True
                    break
            if merged:
                break
    
    # 提取最终结果
    for label, data in region_data.items():
        cx, cy = data['center']
        radius = data['radius']
        radii.append(radius)
        centers.append((cx, cy))
        cv2.circle(img_marked, (cx, cy), int(radius), (0, 255, 0), 2)
    
    num_coins = len(radii)
    avg_radius = np.mean(radii) if radii else 0
    
    print(f"\n分水岭算法检测结果：")
    print(f"硬币数量：{num_coins}")
    print(f"平均半径：{avg_radius:.2f} 像素")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_marked)
    plt.title(f'Watershed: {num_coins} coins')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_7_1_watershed_improved.png', dpi=150)
    plt.show()

def find_best_params():
    """自动寻找最优参数"""
    img = cv2.imread('coins.jpg')
    
    if img is None:
        print("错误：找不到 coins.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist_max = dist_transform.max()
    
    # 测试不同阈值的效果
    ratios = np.arange(0.3, 0.8, 0.05)
    results = []
    
    for ratio in ratios:
        sure_fg = np.uint8(dist_transform > ratio * dist_max)
        num_labels, _ = cv2.connectedComponents(sure_fg)
        coin_count = num_labels - 1
        results.append((ratio, coin_count))
        print(f"阈值比例：{ratio:.2f}，检测数量：{coin_count}")
    
    # 推荐最优参数
    print("\n" + "="*50)
    print("参数推荐：")
    print("阈值比例在0.5-0.6之间通常效果较好")
    print("比例太小会产生过分割（一个硬币分成多个）")
    print("比例太大会产生欠分割（多个硬币粘连在一起）")
    
    return results

if __name__ == "__main__":
    print("实验7-1：硬币图像分割与特征统计（改进版）")
    print("="*50)
    print("1. 基础实验（自动处理粘连和过分割）")
    print("2. 改进分水岭算法")
    print("3. 参数优化建议")
    
    choice = input("请选择 (1/2/3): ")
    
    if choice == "2":
        watershed_with_adaptive_markers()
    elif choice == "3":
        find_best_params()
    else:
        main()