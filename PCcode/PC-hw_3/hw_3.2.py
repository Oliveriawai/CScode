import cv2
import numpy as np
import matplotlib.pyplot as plt

def prewitt_edge_detection(img):
    """Prewitt算子边缘检测（OpenCV没有直接支持，通过卷积实现）"""
    # Prewitt算子模板
    kernel_x = np.array([[-1, 0, 1], 
                         [-1, 0, 1], 
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], 
                         [0, 0, 0], 
                         [1, 1, 1]], dtype=np.float32)
    
    # 卷积运算
    prewitt_x = cv2.filter2D(img, -1, kernel_x)
    prewitt_y = cv2.filter2D(img, -1, kernel_y)
    
    # 计算梯度幅值
    prewitt_edge = cv2.magnitude(prewitt_x.astype(np.float32), 
                                  prewitt_y.astype(np.float32))
    prewitt_edge = np.uint8(np.clip(prewitt_edge, 0, 255))
    
    return prewitt_edge

def sobel_edge_detection(img):
    """Sobel算子边缘检测"""
    # 分别计算x和y方向的梯度
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    sobel_edge = cv2.magnitude(sobel_x, sobel_y)
    sobel_edge = np.uint8(np.clip(sobel_edge, 0, 255))
    
    return sobel_edge

def log_edge_detection(img):
    """LoG算子边缘检测（高斯拉普拉斯）"""
    # 先高斯平滑，再拉普拉斯算子
    # 高斯滤波
    gaussian = cv2.GaussianBlur(img, (5, 5), 1.0)
    # 拉普拉斯算子
    log_edge = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=3)
    log_edge = np.uint8(np.abs(log_edge))
    # 二值化增强显示
    _, log_edge = cv2.threshold(log_edge, 30, 255, cv2.THRESH_BINARY)
    
    return log_edge

def canny_edge_detection(img, low_threshold=50, high_threshold=150):
    """Canny算子边缘检测"""
    # Canny边缘检测（含高斯平滑、梯度计算、非极大值抑制、双阈值检测）
    canny_edge = cv2.Canny(img, low_threshold, high_threshold)
    
    return canny_edge

def main():
    # 1. 读取图像
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取图像，请确认 lena.jpg 文件在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    # 2. 四种边缘检测
    print("正在进行边缘检测...")
    
    # Prewitt算子
    prewitt_edge = prewitt_edge_detection(img)
    
    # Sobel算子
    sobel_edge = sobel_edge_detection(img)
    
    # LoG算子
    log_edge = log_edge_detection(img)
    
    # Canny算子（使用默认阈值）
    canny_edge = canny_edge_detection(img, 50, 150)
    
    # 3. 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('1. Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(prewitt_edge, cmap='gray')
    plt.title('2. Prewitt Operator')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(sobel_edge, cmap='gray')
    plt.title('3. Sobel Operator')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(log_edge, cmap='gray')
    plt.title('4. LoG Operator')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(canny_edge, cmap='gray')
    plt.title('5. Canny Operator')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_3.2_result.png', dpi=150)
    plt.show()
    
    # 4. 输出分析
    print("\n" + "="*50)
    print("边缘检测效果分析")
    print("="*50)
    print("Prewitt算子：边缘较粗，对噪声敏感，计算简单")
    print("Sobel算子：边缘有一定方向性，比Prewitt更精确")
    print("LoG算子：检测过零点，边缘为闭合轮廓，但可能产生双边缘")
    print("Canny算子：边缘连续且细致，抗噪声能力强，效果最好")

def compare_canny_thresholds():
    """比较Canny算子不同阈值的效果"""
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取图像")
        return
    
    # 不同阈值组合
    thresholds = [
        (30, 90, "Low:30, High:90"),
        (50, 150, "Low:50, High:150"),
        (80, 200, "Low:80, High:200"),
        (100, 250, "Low:100, High:250")
    ]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    for i, (low, high, title) in enumerate(thresholds):
        canny_edge = cv2.Canny(img, low, high)
        plt.subplot(2, 3, i+2)
        plt.imshow(canny_edge, cmap='gray')
        plt.title(f'Canny: {title}')
        plt.axis('off')
    
    # 不同sigma值的高斯滤波对Canny的影响
    sigmas = [0.5, 1.0, 1.5, 2.0]
    for i, sigma in enumerate(sigmas):
        blurred = cv2.GaussianBlur(img, (5, 5), sigma)
        canny_edge = cv2.Canny(blurred, 50, 150)
        plt.subplot(2, 4, i+5)
        plt.imshow(canny_edge, cmap='gray')
        plt.title(f'Canny (sigma={sigma})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_3.2_canny_comparison.png', dpi=150)
    plt.show()
    
    print("\n阈值影响分析：")
    print("阈值越低，检测到的边缘越多，但噪声也越多")
    print("阈值越高，检测到的边缘越少，只保留强边缘")
    print("建议高低阈值比例保持在1:2到1:3之间")

def compare_noise_sensitivity():
    """比较各算子对噪声的敏感程度"""
    img_clean = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    img_noisy = cv2.imread('lena_l.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img_clean is None or img_noisy is None:
        print("错误：无法读取图像")
        return
    
    # 对有噪声图像进行边缘检测
    prewitt_noisy = prewitt_edge_detection(img_noisy)
    sobel_noisy = sobel_edge_detection(img_noisy)
    log_noisy = log_edge_detection(img_noisy)
    canny_noisy = canny_edge_detection(img_noisy, 50, 150)
    
    # 显示对比
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(img_clean, cmap='gray')
    plt.title('Clean Image')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(img_noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(prewitt_noisy, cmap='gray')
    plt.title('Prewitt on Noisy')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(sobel_noisy, cmap='gray')
    plt.title('Sobel on Noisy')
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(log_noisy, cmap='gray')
    plt.title('LoG on Noisy')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(canny_noisy, cmap='gray')
    plt.title('Canny on Noisy')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_3.2_noise_comparison.png', dpi=150)
    plt.show()
    
    print("\n噪声敏感性分析：")
    print("Prewitt和Sobel对噪声敏感，噪声图像中会出现大量伪边缘")
    print("LoG由于先进行了高斯平滑，抗噪能力有所提升")
    print("Canny自带高斯平滑和双阈值抑制，抗噪能力最强")

def interactive_canny():
    """交互式调整Canny阈值"""
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取图像")
        return
    
    # 创建窗口
    cv2.namedWindow('Canny Edge Detection')
    
    # 初始化阈值
    low_threshold = 50
    high_threshold = 150
    
    def update(val):
        low = cv2.getTrackbarPos('Low Threshold', 'Canny Edge Detection')
        high = cv2.getTrackbarPos('High Threshold', 'Canny Edge Detection')
        edges = cv2.Canny(img, low, high)
        cv2.imshow('Canny Edge Detection', edges)
    
    # 创建滑动条
    cv2.createTrackbar('Low Threshold', 'Canny Edge Detection', low_threshold, 255, update)
    cv2.createTrackbar('High Threshold', 'Canny Edge Detection', high_threshold, 255, update)
    
    # 显示初始结果
    update(0)
    
    print("\n交互模式说明：")
    print("滑动条调整高低阈值，观察边缘变化")
    print("按ESC键退出")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("实验3.2：图像边缘检测")
    print("="*40)
    print("1. 基础实验（四种算子对比）")
    print("2. Canny算子不同阈值对比")
    print("3. 噪声敏感性对比")
    print("4. 交互式Canny阈值调整")
    
    choice = input("请选择 (1/2/3/4): ")
    
    if choice == "2":
        compare_canny_thresholds()
    elif choice == "3":
        compare_noise_sensitivity()
    elif choice == "4":
        interactive_canny()
    else:
        main()