import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 读取图像
    img = cv2.imread('snowmount.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：找不到 snowmount.jpg")
        print("请将图像文件放在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    # 2. 分割左、中、右三个区域
    h, w = img.shape
    left_region = img[:, :w//3]
    center_region = img[:, w//3:2*w//3]
    right_region = img[:, 2*w//3:]
    
    # 3. 对左区域和右区域进行直方图均衡化
    left_equalized = cv2.equalizeHist(left_region)
    right_equalized = cv2.equalizeHist(right_region)
    
    # 4. 计算直方图
    hist_left = cv2.calcHist([left_region], [0], None, [256], [0, 256])
    hist_center = cv2.calcHist([center_region], [0], None, [256], [0, 256])
    hist_right = cv2.calcHist([right_region], [0], None, [256], [0, 256])
    hist_left_eq = cv2.calcHist([left_equalized], [0], None, [256], [0, 256])
    hist_right_eq = cv2.calcHist([right_equalized], [0], None, [256], [0, 256])
    
    # 5. 显示结果
    plt.figure(figsize=(16, 10))
    
    # 第一行：原始三个区域
    plt.subplot(3, 4, 1)
    plt.imshow(left_region, cmap='gray')
    plt.title('Left Region (Original)')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(center_region, cmap='gray')
    plt.title('Center Region (Original)')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(right_region, cmap='gray')
    plt.title('Right Region (Original)')
    plt.axis('off')
    
    # 第二行：均衡化后的左右区域 + 中间原图对比
    plt.subplot(3, 4, 5)
    plt.imshow(left_equalized, cmap='gray')
    plt.title('Left Region (Equalized)')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(center_region, cmap='gray')
    plt.title('Center Region (Original)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(right_equalized, cmap='gray')
    plt.title('Right Region (Equalized)')
    plt.axis('off')
    
    # 第三行：直方图
    plt.subplot(3, 4, 9)
    plt.plot(hist_left, color='blue', linewidth=1.5)
    plt.title('Left Histogram (Original)')
    plt.xlim([0, 255])
    
    plt.subplot(3, 4, 10)
    plt.plot(hist_center, color='green', linewidth=1.5)
    plt.title('Center Histogram (Original)')
    plt.xlim([0, 255])
    
    plt.subplot(3, 4, 11)
    plt.plot(hist_right, color='red', linewidth=1.5)
    plt.title('Right Histogram (Original)')
    plt.xlim([0, 255])
    
    plt.subplot(3, 4, 12)
    plt.plot(hist_left_eq, color='blue', alpha=0.7, label='Left Eq')
    plt.plot(hist_right_eq, color='red', alpha=0.7, label='Right Eq')
    plt.title('Equalized Histograms')
    plt.legend()
    plt.xlim([0, 255])
    
    plt.tight_layout()
    plt.savefig('hw_4.2_result.png', dpi=150)
    plt.show()
    
    # 6. 统计分析
    print("\n" + "="*50)
    print("直方图均衡化效果分析")
    print("="*50)
    
    print(f"\n左区域均衡化前：均值={np.mean(left_region):.2f}, 标准差={np.std(left_region):.2f}")
    print(f"左区域均衡化后：均值={np.mean(left_equalized):.2f}, 标准差={np.std(left_equalized):.2f}")
    
    print(f"\n右区域均衡化前：均值={np.mean(right_region):.2f}, 标准差={np.std(right_region):.2f}")
    print(f"右区域均衡化后：均值={np.mean(right_equalized):.2f}, 标准差={np.std(right_equalized):.2f}")
    
    print(f"\n中间区域（未处理）：均值={np.mean(center_region):.2f}, 标准差={np.std(center_region):.2f}")

def compare_with_center():
    """将均衡化后的左右区域与中间区域进行对比"""
    img = cv2.imread('snowmount.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：找不到 snowmount.jpg")
        return
    
    h, w = img.shape
    left = img[:, :w//3]
    center = img[:, w//3:2*w//3]
    right = img[:, 2*w//3:]
    
    left_eq = cv2.equalizeHist(left)
    right_eq = cv2.equalizeHist(right)
    
    # 并排显示
    plt.figure(figsize=(15, 8))
    
    # 显示图像
    plt.subplot(2, 4, 1)
    plt.imshow(left, cmap='gray')
    plt.title('Left Original')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(left_eq, cmap='gray')
    plt.title('Left Equalized')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(center, cmap='gray')
    plt.title('Center (Reference)')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(right_eq, cmap='gray')
    plt.title('Right Equalized')
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(right, cmap='gray')
    plt.title('Right Original')
    plt.axis('off')
    
    # 直方图
    plt.subplot(2, 4, 6)
    plt.hist(left.flatten(), bins=50, alpha=0.5, label='Left')
    plt.hist(left_eq.flatten(), bins=50, alpha=0.5, label='Left Eq')
    plt.title('Left Histogram')
    plt.legend()
    
    plt.subplot(2, 4, 7)
    plt.hist(center.flatten(), bins=50, alpha=0.5, label='Center')
    plt.title('Center Histogram')
    plt.legend()
    
    plt.subplot(2, 4, 8)
    plt.hist(right.flatten(), bins=50, alpha=0.5, label='Right')
    plt.hist(right_eq.flatten(), bins=50, alpha=0.5, label='Right Eq')
    plt.title('Right Histogram')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hw_4.2_comparison.png', dpi=150)
    plt.show()
    
    print("\n对比分析：")
    print("均衡化后，左右区域的亮度和对比度更接近中间区域")
    print("说明直方图均衡化可以有效改善图像的曝光一致性")

def adaptive_equalization():
    """对比全局均衡化和自适应均衡化"""
    img = cv2.imread('snowmount.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：找不到 snowmount.jpg")
        return
    
    # 全局直方图均衡化
    global_eq = cv2.equalizeHist(img)
    
    # 自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_eq = clahe.apply(img)
    
    # 显示对比
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(global_eq, cmap='gray')
    plt.title('Global Equalization')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(adaptive_eq, cmap='gray')
    plt.title('Adaptive Equalization (CLAHE)')
    plt.axis('off')
    
    # 直方图
    plt.subplot(2, 3, 4)
    plt.hist(img.flatten(), bins=50, color='gray', alpha=0.7)
    plt.title('Original Histogram')
    
    plt.subplot(2, 3, 5)
    plt.hist(global_eq.flatten(), bins=50, color='gray', alpha=0.7)
    plt.title('Global Equalized Histogram')
    
    plt.subplot(2, 3, 6)
    plt.hist(adaptive_eq.flatten(), bins=50, color='gray', alpha=0.7)
    plt.title('Adaptive Equalized Histogram')
    
    plt.tight_layout()
    plt.savefig('hw_4.2_adaptive.png', dpi=150)
    plt.show()
    
    print("\n全局均衡化 vs 自适应均衡化：")
    print("全局均衡化：整体对比度提升，但可能局部过亮或过暗")
    print("自适应均衡化：局部对比度提升，细节更丰富")

def analyze_cumulative_distribution():
    """分析累积分布函数对均衡化的影响"""
    img = cv2.imread('snowmount.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：找不到 snowmount.jpg")
        return
    
    h, w = img.shape
    left = img[:, :w//3]
    right = img[:, 2*w//3:]
    
    left_eq = cv2.equalizeHist(left)
    right_eq = cv2.equalizeHist(right)
    
    # 计算累积分布
    hist_left = cv2.calcHist([left], [0], None, [256], [0, 256])
    hist_right = cv2.calcHist([right], [0], None, [256], [0, 256])
    
    cdf_left = np.cumsum(hist_left)
    cdf_right = np.cumsum(hist_right)
    cdf_left_norm = cdf_left / cdf_left[-1]
    cdf_right_norm = cdf_right / cdf_right[-1]
    
    # 映射函数
    mapping_left = np.round(cdf_left_norm * 255).astype(np.uint8)
    mapping_right = np.round(cdf_right_norm * 255).astype(np.uint8)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(cdf_left_norm, color='blue')
    plt.title('Left Region CDF')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(cdf_right_norm, color='red')
    plt.title('Right Region CDF')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(mapping_left, color='blue')
    plt.title('Left Mapping Function')
    plt.xlabel('Original Intensity')
    plt.ylabel('Mapped Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(mapping_right, color='red')
    plt.title('Right Mapping Function')
    plt.xlabel('Original Intensity')
    plt.ylabel('Mapped Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hw_4.2_cdf.png', dpi=150)
    plt.show()
    
    print("\n累积分布函数分析：")
    print("CDF斜率大的区域，映射后灰度级被拉伸，对比度增强")
    print("CDF斜率小的区域，映射后灰度级被压缩")

if __name__ == "__main__":
    print("实验4.2：直方图均衡化")
    print("="*40)
    print("1. 基础实验（左右区域均衡化）")
    print("2. 与中间区域对比")
    print("3. 自适应均衡化（CLAHE）")
    print("4. 累积分布函数分析")
    
    choice = input("请选择 (1/2/3/4): ")
    
    if choice == "2":
        compare_with_center()
    elif choice == "3":
        adaptive_equalization()
    elif choice == "4":
        analyze_cumulative_distribution()
    else:
        main()