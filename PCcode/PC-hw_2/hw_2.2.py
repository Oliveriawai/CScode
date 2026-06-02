import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_repeat_image(img_path, repeat_rows=4, repeat_cols=4):
    """
    创建重复图像
    
    参数:
        img_path: 原始图像路径
        repeat_rows: 垂直方向重复次数
        repeat_cols: 水平方向重复次数
    
    返回:
        重复图像
    """
    # 读取原始图像（灰度图）
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"错误：无法读取图像 {img_path}")
        return None
    
    # 获取原始图像尺寸
    h, w = img.shape
    
    print(f"原始图像尺寸：{h} x {w}")
    
    # 创建重复图像
    repeat_img = np.tile(img, (repeat_rows, repeat_cols))
    
    print(f"重复图像尺寸：{repeat_img.shape[0]} x {repeat_img.shape[1]}")
    print(f"重复方式：{repeat_rows} x {repeat_cols} = {repeat_rows * repeat_cols} 个小图")
    
    return repeat_img, img

def fourier_transform_analysis(img, title_prefix=""):
    """
    对图像进行傅里叶变换分析
    """
    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 幅值谱（对数增强）
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1)
    
    # 相位谱
    phase = np.angle(fshift)
    
    return magnitude_log, phase, fshift

def main():
    # 1. 创建重复图像
    print("="*60)
    print("实验2.2：重复图像的傅里叶变换")
    print("="*60)
    
    repeat_img, original_img = create_repeat_image('lena.jpg', repeat_rows=4, repeat_cols=4)
    
    if repeat_img is None:
        return
    
    # 2. 对重复图像进行傅里叶变换
    magnitude_log, phase, fshift = fourier_transform_analysis(repeat_img)
    
    # 3. 为了对比，也对原始单张图像进行傅里叶变换
    original_magnitude_log, original_phase, _ = fourier_transform_analysis(original_img)
    
    # 4. 显示结果
    plt.figure(figsize=(16, 12))
    
    # 第一行：原始图像和重复图像
    plt.subplot(2, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('1. Original Single Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(repeat_img, cmap='gray')
    plt.title(f'2. Repeated Image (4x4 = 16 copies)')
    plt.axis('off')
    
    # 第二行：原始图像的频谱
    plt.subplot(2, 3, 3)
    plt.imshow(original_magnitude_log, cmap='gray')
    plt.title('3. Original Image - Magnitude Spectrum')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(original_phase, cmap='gray')
    plt.title('4. Original Image - Phase Spectrum')
    plt.axis('off')
    
    # 第三行：重复图像的频谱
    plt.subplot(2, 3, 5)
    plt.imshow(magnitude_log, cmap='gray')
    plt.title('5. Repeated Image - Magnitude Spectrum (Notice the grid pattern!)')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(2, 3, 6)
    plt.imshow(phase, cmap='gray')
    plt.title('6. Repeated Image - Phase Spectrum')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_2.2_result.png', dpi=150)
    plt.show()
    
    # 5. 分析输出
    print("\n" + "="*60)
    print("观察与分析")
    print("="*60)
    
    print("\n【与实验2.1的相同点】")
    print("1. 幅值谱中心仍然最亮（低频分量能量最大）")
    print("2. 相位谱看起来仍像随机噪声")
    print("3. 幅值谱仍然对称（傅里叶变换的共轭对称性）")
    
    print("\n【与实验2.1的不同点】")
    print("1. 重复图像的幅值谱出现了明显的网格状亮点！")
    print("2. 这些亮点对应重复图案的周期性结构")
    print("3. 亮点间距与重复周期有关")
    
    print("\n【原因解释】")
    print("• 当图像中重复出现相同图案时，相当于在空间域进行了周期性延拓")
    print("• 在频域中，这种周期性会表现为离散的峰值点")
    print("• 4x4重复 → 频谱中出现4x4的网格状结构")
    print("• 这符合傅里叶变换的卷积定理：时域周期化对应频域离散化")
    
    print("\n【观察要点】")
    print("• 注意幅值谱中的十字亮线仍然存在（来自Lena图像的边缘）")
    print("• 但叠加了周期性的网格结构（来自重复排列）")
    print("• 相位谱受重复影响较小，仍保留结构信息")

def visualize_fft_grid_effect():
    """
    可视化展示不同重复次数对频谱的影响
    """
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取 lena.jpg")
        return
    
    repeat_configs = [
        (1, 1, "1x1 (Original)"),
        (2, 2, "2x2"),
        (3, 3, "3x3"),
        (4, 4, "4x4")
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, (rows, cols, title) in enumerate(repeat_configs):
        # 创建重复图像
        repeat_img = np.tile(img, (rows, cols))
        
        # 傅里叶变换
        f = np.fft.fft2(repeat_img)
        fshift = np.fft.fftshift(f)
        magnitude_log = np.log(np.abs(fshift) + 1)
        
        # 显示空间域图像
        axes[0, idx].imshow(repeat_img, cmap='gray')
        axes[0, idx].set_title(f'Spatial: {title}', fontsize=10)
        axes[0, idx].axis('off')
        
        # 显示频域幅值谱
        axes[1, idx].imshow(magnitude_log, cmap='gray')
        axes[1, idx].set_title(f'Frequency: {title}', fontsize=10)
        axes[1, idx].axis('off')
    
    plt.suptitle('Effect of Repetition on Fourier Spectrum', fontsize=14)
    plt.tight_layout()
    plt.savefig('hw_2.2_comparison.png', dpi=150)
    plt.show()
    
    print("\n结论：重复次数越多，频谱中的网格状结构越明显！")

def interactive_explanation():
    """
    交互式显示，带标注解释
    """
    repeat_img, original_img = create_repeat_image('lena.jpg', 4, 4)
    
    if repeat_img is None:
        return
    
    # 傅里叶变换
    f = np.fft.fft2(repeat_img)
    fshift = np.fft.fftshift(f)
    magnitude_log = np.log(np.abs(fshift) + 1)
    
    # 归一化
    magnitude_display = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 在频谱上标记网格点
    magnitude_color = cv2.cvtColor(magnitude_display, cv2.COLOR_GRAY2BGR)
    
    h, w = magnitude_display.shape
    cy, cx = h // 2, w // 2
    
    # 标记中心点和网格点
    step_y = h // 8  # 4x4重复对应的频谱间隔
    step_x = w // 8
    
    for i in range(-3, 4):
        for j in range(-3, 4):
            y = cy + i * step_y
            x = cx + j * step_x
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(magnitude_color, (x, y), 3, (0, 0, 255), -1)
    
    # 显示
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(repeat_img, cmap='gray')
    plt.title('Spatial Domain: 4x4 Repeated Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_log, cmap='gray')
    plt.title('Frequency Domain: Magnitude Spectrum')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(magnitude_color)
    plt.title('Marked Grid Points (Red circles)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_2.2_annotated.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("图解说明")
    print("="*60)
    print("红色圆圈标记了频谱中出现的网格状亮点位置")
    print("这些亮点对应于空间域中4x4重复图案的周期性")
    print("\n物理意义：")
    print("• 中心点：直流分量（图像平均亮度）")
    print("• 周围网格点：对应重复频率的谐波分量")
    print("• 网格间距 ∝ 1 / 重复周期")

if __name__ == "__main__":
    print("请选择运行模式：")
    print("1. 基础实验（4x4重复图像傅里叶变换）")
    print("2. 对比不同重复次数的影响")
    print("3. 交互式标注解释")
    
    choice = input("请输入选择 (1/2/3): ")
    
    if choice == "2":
        visualize_fft_grid_effect()
    elif choice == "3":
        interactive_explanation()
    else:
        main()