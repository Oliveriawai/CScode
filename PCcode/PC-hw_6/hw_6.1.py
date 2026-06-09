import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 读取图像
    img = cv2.imread('building.jpg')
    
    if img is None:
        print("错误：找不到 building.jpg")
        print("请将图像文件放在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    # 2. 转换为灰度图并进行边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用Canny边缘检测（霍夫变换的预处理）
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 3. 标准霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    # 4. 在图像上绘制检测到的直线
    img_lines = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if lines is not None:
        print(f"检测到的直线数量（标准霍夫）：{len(lines)}")
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("未检测到直线，请调整阈值参数")
    
    # 5. 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.title(f'Hough Lines Detection (Standard, {len(lines) if lines is not None else 0} lines)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.1_result.png', dpi=150)
    plt.show()
    
    return edges

def probabilistic_hough():
    """概率霍夫变换（更常用，直接返回线段端点）"""
    img = cv2.imread('building.jpg')
    
    if img is None:
        print("错误：找不到 building.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 概率霍夫变换
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    
    img_lines = img.copy()
    img_rgb = cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)
    
    if lines is not None:
        print(f"检测到的线段数量（概率霍夫）：{len(lines)}")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    else:
        print("未检测到线段")
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_rgb)
    plt.title(f'Probabilistic Hough Lines ({len(lines) if lines is not None else 0} segments)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.1_probabilistic.png', dpi=150)
    plt.show()

def compare_parameters():
    """比较不同参数对霍夫变换结果的影响"""
    img = cv2.imread('building.jpg')
    
    if img is None:
        print("错误：找不到 building.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 不同阈值参数
    thresholds = [100, 150, 200, 250]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Canny Edges')
    axes[0, 1].axis('off')
    
    for i, thresh in enumerate(thresholds):
        lines = cv2.HoughLines(edges, 1, np.pi/180, thresh)
        
        img_lines = img.copy()
        img_rgb = cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            axes[1, i].imshow(img_rgb)
            axes[1, i].set_title(f'Threshold = {thresh}\n({len(lines)} lines)')
        else:
            axes[1, i].imshow(img_rgb)
            axes[1, i].set_title(f'Threshold = {thresh}\n(No lines)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.1_compare_threshold.png', dpi=150)
    plt.show()
    
    print("\n阈值参数影响分析：")
    print("阈值越低，检测到的直线越多，但可能包含噪声")
    print("阈值越高，只检测明显的直线，但可能漏检")
    print("建议阈值范围：150-250")

def compare_probabilistic_params():
    """比较概率霍夫变换的不同参数"""
    img = cv2.imread('building.jpg')
    
    if img is None:
        print("错误：找不到 building.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 不同参数组合
    param_sets = [
        (100, 30, 5, "threshold=100, minLen=30, gap=5"),
        (100, 50, 10, "threshold=100, minLen=50, gap=10"),
        (150, 50, 10, "threshold=150, minLen=50, gap=10"),
        (150, 80, 20, "threshold=150, minLen=80, gap=20"),
        (200, 50, 10, "threshold=200, minLen=50, gap=10"),
        (200, 100, 20, "threshold=200, minLen=100, gap=20"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Canny Edges')
    axes[0, 1].axis('off')
    
    for i, (thresh, min_len, gap, title) in enumerate(param_sets):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, thresh, minLineLength=min_len, maxLineGap=gap)
        
        img_lines = img.copy()
        img_rgb = cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            row = (i + 2) // 3
            col = (i + 2) % 3
            axes[row, col].imshow(img_rgb)
            axes[row, col].set_title(f'{len(lines)} lines\n{title[:30]}')
        else:
            row = (i + 2) // 3
            col = (i + 2) % 3
            axes[row, col].imshow(img_rgb)
            axes[row, col].set_title(f'No lines\n{title[:30]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.1_compare_prob_params.png', dpi=150)
    plt.show()
    
    print("\n概率霍夫变换参数影响：")
    print("threshold：累加器阈值，越大要求直线越明显")
    print("minLineLength：最小线段长度，过滤短线段")
    print("maxLineGap：线段间最大间隔，用于连接断开的线段")

def detect_vertical_horizontal():
    """分别检测垂直和水平直线"""
    img = cv2.imread('building.jpg')
    
    if img is None:
        print("错误：找不到 building.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    
    vertical_lines = []
    horizontal_lines = []
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # 判断直线方向（角度）
            angle = theta * 180 / np.pi
            
            if (angle > 80 and angle < 100) or (angle > 260 and angle < 280):
                vertical_lines.append(line)
            elif (angle < 10 or angle > 350) or (angle > 170 and angle < 190):
                horizontal_lines.append(line)
    
    # 绘制结果
    img_vertical = img.copy()
    img_horizontal = img.copy()
    img_all = img.copy()
    
    img_vertical_rgb = cv2.cvtColor(img_vertical, cv2.COLOR_BGR2RGB)
    img_horizontal_rgb = cv2.cvtColor(img_horizontal, cv2.COLOR_BGR2RGB)
    img_all_rgb = cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB)
    
    # 绘制垂直线（红色）
    for line in vertical_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_vertical_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.line(img_all_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # 绘制水平线（蓝色）
    for line in horizontal_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_horizontal_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(img_all_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 显示
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(img_vertical_rgb)
    plt.title(f'Vertical Lines ({len(vertical_lines)})')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_horizontal_rgb)
    plt.title(f'Horizontal Lines ({len(horizontal_lines)})')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(img_all_rgb)
    plt.title(f'All Lines (V:{len(vertical_lines)}, H:{len(horizontal_lines)})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.1_directions.png', dpi=150)
    plt.show()
    
    print(f"\n检测结果：垂直线{len(vertical_lines)}条，水平线{len(horizontal_lines)}条")

def interactive_hough():
    """交互式霍夫变换参数调整"""
    img = cv2.imread('building.jpg')
    
    if img is None:
        print("错误：找不到 building.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 窗口
    cv2.namedWindow('Hough Lines')
    cv2.namedWindow('Edges')
    cv2.imshow('Edges', edges)
    
    # 初始化参数
    threshold = 150
    
    def update(val):
        nonlocal threshold
        threshold = cv2.getTrackbarPos('Threshold', 'Hough Lines')
        
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
        
        img_lines = img.copy()
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.putText(img_lines, f'Threshold: {threshold}, Lines: {len(lines) if lines is not None else 0}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Hough Lines', img_lines)
    
    cv2.createTrackbar('Threshold', 'Hough Lines', threshold, 300, update)
    update(threshold)
    
    print("\n交互模式说明：")
    print("调整滑动条改变阈值，观察直线检测变化")
    print("按ESC键退出")
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("实验6.1：Hough变换直线检测")
    print("="*50)
    print("1. 基础实验（标准霍夫变换）")
    print("2. 概率霍夫变换（线段检测）")
    print("3. 比较不同阈值参数")
    print("4. 比较概率霍夫不同参数")
    print("5. 分别检测垂直和水平线")
    print("6. 交互式参数调整")
    
    choice = input("请选择 (1/2/3/4/5/6): ")
    
    if choice == "2":
        probabilistic_hough()
    elif choice == "3":
        compare_parameters()
    elif choice == "4":
        compare_probabilistic_params()
    elif choice == "5":
        detect_vertical_horizontal()
    elif choice == "6":
        interactive_hough()
    else:
        main()