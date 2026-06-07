import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. 读取图像（带噪声的图像）
    img = cv2.imread('lena_l.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取图像，请确认 lena_l.jpg 文件在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    # 2. 三种滤波处理
    # 平均滤波：用邻域内所有像素的平均值替换中心像素
    blur_mean = cv2.blur(img, (5, 5))
    
    # 高斯滤波：根据距离赋予不同权重，中心权重最大
    blur_gaussian = cv2.GaussianBlur(img, (5, 5), 1.5)
    
    # 中值滤波：用邻域内像素的中值替换中心像素，对椒盐噪声效果好
    blur_median = cv2.medianBlur(img, 5)
    
    # 3. 显示结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('1. Original Image (with noise)')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(blur_mean, cmap='gray')
    plt.title('2. Mean Filter (5x5)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(blur_gaussian, cmap='gray')
    plt.title('3. Gaussian Filter (5x5, sigma=1.5)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(blur_median, cmap='gray')
    plt.title('4. Median Filter (5x5)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_3.1_result.png', dpi=150)
    plt.show()
    
    # 4. 输出分析
    print("\n" + "="*50)
    print("滤波效果分析")
    print("="*50)
    print("平均滤波：图像整体变模糊，边缘不清晰")
    print("高斯滤波：过渡更自然，边缘保留相对较好")
    print("中值滤波：去噪效果好，边缘清晰度最高")

def compare_kernel_sizes():
    """比较不同核大小对滤波效果的影响"""
    img = cv2.imread('lena_l.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取 lena_l.jpg")
        return
    
    kernel_sizes = [3, 5, 7, 9]
    
    fig, axes = plt.subplots(3, len(kernel_sizes) + 1, figsize=(16, 10))
    
    # 原图
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(img, cmap='gray')
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')
    axes[2, 0].imshow(img, cmap='gray')
    axes[2, 0].set_title('Original')
    axes[2, 0].axis('off')
    
    for i, k in enumerate(kernel_sizes):
        # 平均滤波
        blur_mean = cv2.blur(img, (k, k))
        axes[0, i+1].imshow(blur_mean, cmap='gray')
        axes[0, i+1].set_title(f'Mean {k}x{k}')
        axes[0, i+1].axis('off')
        
        # 高斯滤波
        blur_gaussian = cv2.GaussianBlur(img, (k, k), 0)
        axes[1, i+1].imshow(blur_gaussian, cmap='gray')
        axes[1, i+1].set_title(f'Gaussian {k}x{k}')
        axes[1, i+1].axis('off')
        
        # 中值滤波
        blur_median = cv2.medianBlur(img, k)
        axes[2, i+1].imshow(blur_median, cmap='gray')
        axes[2, i+1].set_title(f'Median {k}x{k}')
        axes[2, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_3.1_kernel_comparison.png', dpi=150)
    plt.show()

def compare_with_original():
    """对比滤波结果与原图lena.jpg的差异"""
    img_noisy = cv2.imread('lena_l.jpg', cv2.IMREAD_GRAYSCALE)
    img_clean = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img_noisy is None or img_clean is None:
        print("错误：无法读取图像")
        return
    
    # 滤波
    blur_mean = cv2.blur(img_noisy, (5, 5))
    blur_gaussian = cv2.GaussianBlur(img_noisy, (5, 5), 1.5)
    blur_median = cv2.medianBlur(img_noisy, 5)
    
    # 计算PSNR（峰值信噪比）
    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    print("\n与原始无噪图像对比（PSNR值，越大越好）：")
    print(f"平均滤波后：{psnr(blur_mean, img_clean):.2f} dB")
    print(f"高斯滤波后：{psnr(blur_gaussian, img_clean):.2f} dB")
    print(f"中值滤波后：{psnr(blur_median, img_clean):.2f} dB")

if __name__ == "__main__":
    print("实验3.1：图像平滑滤波")
    print("1. 基础实验")
    print("2. 不同核大小对比")
    print("3. 定量对比（PSNR）")
    
    choice = input("请选择 (1/2/3): ")
    
    if choice == "2":
        compare_kernel_sizes()
    elif choice == "3":
        compare_with_original()
    else:
        main()