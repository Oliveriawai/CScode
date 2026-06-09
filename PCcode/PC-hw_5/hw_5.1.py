import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 读取图像
    img = cv2.imread('rice.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：找不到 rice.jpg")
        print("请将图像文件放在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    # 2. 大津阈值分割
    ret, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"大津算法计算的最优阈值：{ret:.2f}")
    
    # 3. 显示结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image (rice.jpg)')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.hist(img.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.axvline(x=ret, color='red', linestyle='--', linewidth=2, label=f'Threshold={ret:.0f}')
    plt.title('Gray Histogram with Otsu Threshold')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.imshow(thresh_otsu, cmap='gray')
    plt.title(f'Otsu Threshold Segmentation (threshold={ret:.0f})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_5.1_result.png', dpi=150)
    plt.show()
    
    return img, thresh_otsu, ret

def analyze_rice_grains(img, thresh):
    """计算米粒的平均面积和长度"""
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n检测到的米粒数量：{len(contours)}")
    
    # 存储每个米粒的面积和长度
    areas = []
    lengths = []
    
    for contour in contours:
        # 计算面积
        area = cv2.contourArea(contour)
        if area > 10:  # 忽略太小的噪声区域
            areas.append(area)
            
            # 计算最小外接矩形，获取长度
            rect = cv2.minAreaRect(contour)
            length = max(rect[1])  # 取较长的一边作为长度
            lengths.append(length)
    
    # 去除异常值后计算平均值
    areas = np.array(areas)
    lengths = np.array(lengths)
    
    # 排除过大或过小的异常值（使用四分位距）
    if len(areas) > 5:
        q1_area = np.percentile(areas, 25)
        q3_area = np.percentile(areas, 75)
        iqr_area = q3_area - q1_area
        areas_filtered = areas[(areas >= q1_area - 1.5*iqr_area) & (areas <= q3_area + 1.5*iqr_area)]
        
        q1_len = np.percentile(lengths, 25)
        q3_len = np.percentile(lengths, 75)
        iqr_len = q3_len - q1_len
        lengths_filtered = lengths[(lengths >= q1_len - 1.5*iqr_len) & (lengths <= q3_len + 1.5*iqr_len)]
    else:
        areas_filtered = areas
        lengths_filtered = lengths
    
    avg_area = np.mean(areas_filtered) if len(areas_filtered) > 0 else 0
    avg_length = np.mean(lengths_filtered) if len(lengths_filtered) > 0 else 0
    std_area = np.std(areas_filtered) if len(areas_filtered) > 0 else 0
    std_length = np.std(lengths_filtered) if len(lengths_filtered) > 0 else 0
    
    print(f"\n米粒面积统计：")
    print(f"  平均面积：{avg_area:.2f} 像素")
    print(f"  面积标准差：{std_area:.2f}")
    print(f"  最小面积：{np.min(areas_filtered):.2f}")
    print(f"  最大面积：{np.max(areas_filtered):.2f}")
    
    print(f"\n米粒长度统计：")
    print(f"  平均长度：{avg_length:.2f} 像素")
    print(f"  长度标准差：{std_length:.2f}")
    print(f"  最小长度：{np.min(lengths_filtered):.2f}")
    print(f"  最大长度：{np.max(lengths_filtered):.2f}")
    
    return contours, areas, lengths

def visualize_grains(img, thresh, contours):
    """可视化米粒检测结果"""
    # 在原图上标记米粒轮廓
    img_contour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)
    
    # 标记每个米粒的编号和面积
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 10:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(img_contour, f'{i+1}', (cx-10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # 绘制面积分布直方图
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_contour)
    plt.title(f'Detected Rice Grains (Total: {len(contours)})')
    plt.axis('off')
    
    # 计算面积分布
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
    
    plt.subplot(1, 2, 2)
    plt.hist(areas, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.title('Rice Grain Area Distribution')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Number of Grains')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hw_5.1_grains.png', dpi=150)
    plt.show()

def compare_threshold_methods():
    """对比不同阈值分割方法的效果"""
    img = cv2.imread('rice.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：找不到 rice.jpg")
        return
    
    # 不同分割方法
    # 1. 固定阈值
    ret_fixed, thresh_fixed = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    
    # 2. 大津算法
    ret_otsu, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. 自适应阈值
    thresh_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    
    # 显示对比
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(thresh_fixed, cmap='gray')
    plt.title(f'Fixed Threshold (T=120)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(thresh_otsu, cmap='gray')
    plt.title(f'Otsu Threshold (T={ret_otsu:.0f})')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(thresh_adaptive, cmap='gray')
    plt.title('Adaptive Threshold (Gaussian)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_5.1_comparison.png', dpi=150)
    plt.show()
    
    print("\n不同阈值方法对比：")
    print("固定阈值：简单但需要人工调整，对不同光照效果差")
    print("大津算法：自动确定全局最优阈值，适合双峰直方图")
    print("自适应阈值：局部阈值，适合光照不均匀的图像")

def improve_segmentation(img):
    """尝试改善分割效果"""
    # 先进行高斯滤波去噪
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    
    # 形态学操作：开运算去除小噪声
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 大津阈值
    ret, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 闭运算填充空洞
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 显示改善效果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(blur, cmap='gray')
    plt.title('After Gaussian Blur')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(morph, cmap='gray')
    plt.title('After Morphology (Open)')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(thresh, cmap='gray')
    plt.title('Improved Segmentation')
    plt.axis('off')
    
    # 对比原分割结果
    _, thresh_original = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.subplot(2, 3, 5)
    plt.imshow(thresh_original, cmap='gray')
    plt.title('Original Otsu Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_5.1_improved.png', dpi=150)
    plt.show()
    
    # 统计改善后的米粒
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
    print(f"\n改善后检测到的米粒数量：{len(valid_contours)}")
    
    return thresh

def interactive_threshold():
    """交互式调整阈值"""
    img = cv2.imread('rice.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：找不到 rice.jpg")
        return
    
    cv2.namedWindow('Threshold Adjustment')
    
    def update(threshold):
        _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('Threshold Adjustment', thresh)
    
    # 计算大津阈值作为参考
    ret_otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.createTrackbar('Threshold', 'Threshold Adjustment', int(ret_otsu), 255, update)
    update(ret_otsu)
    
    print(f"\n交互模式说明：")
    print(f"大津算法推荐阈值：{ret_otsu:.0f}")
    print("调整滑动条观察不同阈值的分割效果")
    print("按ESC键退出")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("实验5.1：大津算法阈值分割")
    print("="*40)
    print("1. 基础实验（大津阈值分割）")
    print("2. 米粒面积和长度统计")
    print("3. 不同阈值方法对比")
    print("4. 改善分割效果（形态学处理）")
    print("5. 交互式阈值调整")
    
    choice = input("请选择 (1/2/3/4/5): ")
    
    if choice == "2":
        img, thresh, ret = main()
        contours, areas, lengths = analyze_rice_grains(img, thresh)
        visualize_grains(img, thresh, contours)
    elif choice == "3":
        compare_threshold_methods()
    elif choice == "4":
        img = cv2.imread('rice.jpg', cv2.IMREAD_GRAYSCALE)
        if img is not None:
            improve_segmentation(img)
    elif choice == "5":
        interactive_threshold()
    else:
        main()