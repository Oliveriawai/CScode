import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 检查当前目录和图像文件
    print(f"当前工作目录：{os.getcwd()}")
    
    # 列出当前目录下的jpg文件
    files = [f for f in os.listdir('.') if f.endswith('.jpg') or f.endswith('.png')]
    print(f"当前目录下的图像文件：{files}")
    
    # 2. 读取图像（尝试多个可能的文件名）
    img = None
    possible_names = ['snowmount.jpg', 'snowmount.png', 'snow.jpg', 'test.jpg']
    
    for name in possible_names:
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"成功读取图像：{name}")
            break
    
    if img is None:
        print("错误：找不到图像文件")
        print("请确保 snowmount.jpg 在当前目录下")
        print(f"当前目录：{os.getcwd()}")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    # 3. 分割左、中、右三个区域
    h, w = img.shape
    left_region = img[:, :w//3]
    center_region = img[:, w//3:2*w//3]
    right_region = img[:, 2*w//3:]
    
    # 4. 计算三个区域的直方图
    hist_left = cv2.calcHist([left_region], [0], None, [256], [0, 256])
    hist_center = cv2.calcHist([center_region], [0], None, [256], [0, 256])
    hist_right = cv2.calcHist([right_region], [0], None, [256], [0, 256])
    
    # 5. 显示图像和直方图
    plt.figure(figsize=(14, 8))
    
    # 显示左中右三个区域
    plt.subplot(2, 3, 1)
    plt.imshow(left_region, cmap='gray')
    plt.title('Left Region')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(center_region, cmap='gray')
    plt.title('Center Region')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(right_region, cmap='gray')
    plt.title('Right Region')
    plt.axis('off')
    
    # 绘制三个直方图
    plt.subplot(2, 3, 4)
    plt.plot(hist_left, color='blue', linewidth=1.5)
    plt.title('Left Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    
    plt.subplot(2, 3, 5)
    plt.plot(hist_center, color='green', linewidth=1.5)
    plt.title('Center Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    
    plt.subplot(2, 3, 6)
    plt.plot(hist_right, color='red', linewidth=1.5)
    plt.title('Right Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    
    plt.tight_layout()
    plt.savefig('hw_4.1_result.png', dpi=150)
    plt.show()
    
    # 6. 输出统计分析
    print("\n" + "="*50)
    print("灰度直方图统计分析")
    print("="*50)
    
    print(f"\n左区域：均值={np.mean(left_region):.2f}, 标准差={np.std(left_region):.2f}")
    print(f"中区域：均值={np.mean(center_region):.2f}, 标准差={np.std(center_region):.2f}")
    print(f"右区域：均值={np.mean(right_region):.2f}, 标准差={np.std(right_region):.2f}")
    
    # 峰值位置
    print(f"\n左区域直方图峰值位置：{np.argmax(hist_left)}")
    print(f"中区域直方图峰值位置：{np.argmax(hist_center)}")
    print(f"右区域直方图峰值位置：{np.argmax(hist_right)}")

if __name__ == "__main__":
    main()