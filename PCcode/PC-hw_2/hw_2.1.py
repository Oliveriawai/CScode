import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(img):
    """
    对图像进行傅里叶变换，返回幅值谱和相位谱
    """
    # 1. 进行二维傅里叶变换
    f = np.fft.fft2(img)
    
    # 2. 将低频分量移到频谱中心
    fshift = np.fft.fftshift(f)
    
    # 3. 计算幅值谱（取绝对值，并取对数增强显示）
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1)  # +1防止log(0)
    
    # 4. 计算相位谱
    phase = np.angle(fshift)  # 范围 -π 到 π
    
    return fshift, magnitude_log, phase

def reconstruct_from_magnitude_phase(magnitude, phase):
    """
    从幅值谱和相位谱重构图像
    """
    # 组合复数： magnitude * e^(j*phase)
    complex_img = magnitude * np.exp(1j * phase)
    
    # 逆傅里叶变换
    img_back = np.real(np.fft.ifft2(complex_img))
    
    return img_back

def main():
    # 1. 读取图像（转为灰度图）
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取图像，请确认 lena.jpg 文件在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape}")
    print(f"图像数据类型：{img.dtype}")
    
    # 2. 显示原图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('1. Original Image')
    plt.axis('off')
    
    # 3. 傅里叶变换
    fshift, magnitude_log, phase = fourier_transform(img)
    
    # 4. 显示幅值谱
    plt.subplot(2, 3, 2)
    plt.imshow(magnitude_log, cmap='gray')
    plt.title('2. Magnitude Spectrum (Log Scale)')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # 5. 显示相位谱
    plt.subplot(2, 3, 3)
    plt.imshow(phase, cmap='gray')
    plt.title('3. Phase Spectrum')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # 6. 验证：仅用幅值重构（相位设为0）
    magnitude_only = np.abs(fshift)
    recon_magnitude_only = reconstruct_from_magnitude_phase(magnitude_only, 0)
    plt.subplot(2, 3, 4)
    plt.imshow(recon_magnitude_only, cmap='gray')
    plt.title('4. Reconstructed from Magnitude Only')
    plt.axis('off')
    
    # 7. 验证：仅用相位重构（幅值设为1）
    phase_only = np.angle(fshift)
    recon_phase_only = reconstruct_from_magnitude_phase(1, phase_only)
    plt.subplot(2, 3, 5)
    plt.imshow(recon_phase_only, cmap='gray')
    plt.title('5. Reconstructed from Phase Only')
    plt.axis('off')
    
    # 8. 显示频谱中心放大区域（观察细节）
    h, w = img.shape
    cy, cx = h // 2, w // 2
    zoom_size = 50
    zoom_magnitude = magnitude_log[cy-zoom_size:cy+zoom_size, cx-zoom_size:cx+zoom_size]
    
    plt.subplot(2, 3, 6)
    plt.imshow(zoom_magnitude, cmap='gray')
    plt.title('6. Zoomed Center of Magnitude Spectrum')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_2.1_result.png', dpi=150)
    plt.show()
    
    # 9. 输出分析结果
    print("\n" + "="*50)
    print("分析结果：")
    print("="*50)
    print("1. 幅值谱特点：")
    print("   - 中心最亮（低频分量）")
    print("   - 从中心向外逐渐变暗（高频分量减弱）")
    print("   - 十字亮线（图像边缘的方向性）")
    print("\n2. 相位谱特点：")
    print("   - 看起来像随机噪声")
    print("   - 但包含了图像的结构信息")
    print("\n3. 重要结论：")
    print("   - 从幅值谱几乎看不出原始图像")
    print("   - 从相位谱可以大致看出原始图像的轮廓！")
    print("   - 仅用相位重构的图像（图5）能看出Lena的轮廓")
    print("   - 仅用幅值重构的图像（图4）看不到任何结构")
    print("\n原因：相位谱保留了图像的位置和结构信息，")
    print("      幅值谱只保留了各频率成分的能量强度。")

def fourier_analysis_detailed():
    """详细分析模式（OpenCV窗口）"""
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取 lena.jpg")
        return
    
    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 幅值谱
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1)
    
    # 相位谱
    phase = np.angle(fshift)
    
    # 归一化到0-255范围便于显示
    magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
    phase_norm = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)
    
    # 创建显示窗口
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Magnitude Spectrum', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Phase Spectrum', cv2.WINDOW_NORMAL)
    
    # 调整窗口大小
    cv2.resizeWindow('Original', 400, 400)
    cv2.resizeWindow('Magnitude Spectrum', 400, 400)
    cv2.resizeWindow('Phase Spectrum', 400, 400)
    
    # 显示
    cv2.imshow('Original', img)
    cv2.imshow('Magnitude Spectrum', magnitude_norm.astype(np.uint8))
    cv2.imshow('Phase Spectrum', phase_norm.astype(np.uint8))
    
    print("\n" + "="*60)
    print("傅里叶变换分析")
    print("="*60)
    print("\n【幅值谱 Magnitude Spectrum】")
    print("- 中心亮点：代表图像的低频分量（平滑区域）")
    print("- 周围区域：代表高频分量（边缘、纹理）")
    print("- 十字亮线：由图像的水平和垂直边缘产生")
    print("\n【相位谱 Phase Spectrum】")
    print("- 看起来像噪声，实际包含图像的位置信息")
    print("- 决定了图像中物体的形状和轮廓")
    print("\n【结论】")
    print("✓ 从相位谱能大致看出原始图像的形状！")
    print("✓ 幅值谱只能看出能量分布，看不出图像内容")
    
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_with_modified():
    """对比实验：原图 vs 仅相位重构 vs 仅幅值重构"""
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取 lena.jpg")
        return
    
    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    # 仅用相位重构（幅值设为常数1）
    recon_phase = np.real(np.fft.ifft2(np.fft.ifftshift(np.exp(1j * phase))))
    
    # 仅用幅值重构（相位设为0）
    recon_magnitude = np.real(np.fft.ifft2(np.fft.ifftshift(magnitude)))
    
    # 归一化显示
    recon_phase_norm = cv2.normalize(recon_phase, None, 0, 255, cv2.NORM_MINMAX)
    recon_magnitude_norm = cv2.normalize(recon_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # 并排显示
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(recon_phase_norm, cmap='gray')
    plt.title('Reconstructed from Phase Only\n(Can see the outline!)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(recon_magnitude_norm, cmap='gray')
    plt.title('Reconstructed from Magnitude Only\n(Cannot see the shape)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_2.1_comparison.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("请选择运行模式：")
    print("1. 基础傅里叶变换显示（matplotlib完整分析）")
    print("2. 详细分析（OpenCV窗口）")
    print("3. 对比实验（仅相位 vs 仅幅值）")
    
    choice = input("请输入选择 (1/2/3): ")
    
    if choice == "2":
        fourier_analysis_detailed()
    elif choice == "3":
        compare_with_modified()
    else:
        main()