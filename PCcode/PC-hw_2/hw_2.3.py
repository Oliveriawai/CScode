import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(img):
    """傅里叶变换，返回频域数据"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def inverse_fourier_transform(fshift):
    """逆傅里叶变换，返回空间域图像"""
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.real(np.fft.ifft2(f_ishift))
    return img_back

def apply_gaussian_to_magnitude(magnitude, sigma=20):
    """对幅值谱进行高斯滤波"""
    magnitude_filtered = cv2.GaussianBlur(magnitude, (0, 0), sigma)
    
    # 保持直流分量不变（中心点）
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    magnitude_filtered[cy, cx] = magnitude[cy, cx]
    
    return magnitude_filtered

def apply_gaussian_to_phase(phase, sigma=20):
    """对相位谱进行高斯滤波"""
    # 将相位转换为复数形式便于滤波
    phase_complex = np.exp(1j * phase)
    
    # 对实部和虚部分别滤波
    real_part = np.real(phase_complex)
    imag_part = np.imag(phase_complex)
    
    real_filtered = cv2.GaussianBlur(real_part, (0, 0), sigma)
    imag_filtered = cv2.GaussianBlur(imag_part, (0, 0), sigma)
    
    # 重新计算相位
    phase_filtered = np.angle(real_filtered + 1j * imag_filtered)
    
    return phase_filtered

def reconstruct_from_components(magnitude, phase):
    """从幅值和相位重构图像"""
    complex_img = magnitude * np.exp(1j * phase)
    img_back = inverse_fourier_transform(complex_img)
    # 归一化到0-255
    img_back = np.clip(img_back, 0, 255)
    return img_back.astype(np.uint8)

def main():
    # 1. 读取图像
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取 lena.jpg")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    # 2. 傅里叶变换
    fshift = fourier_transform(img)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    # 保存原始幅值和相位
    magnitude_original = magnitude.copy()
    phase_original = phase.copy()
    
    # 3. 显示原始图像和频谱
    plt.figure(figsize=(16, 12))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('1. Original Image')
    plt.axis('off')
    
    # 原始幅值谱（对数显示）
    magnitude_log = np.log(magnitude + 1)
    plt.subplot(2, 3, 2)
    plt.imshow(magnitude_log, cmap='gray')
    plt.title('2. Original Magnitude Spectrum')
    plt.axis('off')
    
    # 原始相位谱
    plt.subplot(2, 3, 3)
    plt.imshow(phase, cmap='gray')
    plt.title('3. Original Phase Spectrum')
    plt.axis('off')
    
    # ========== 实验①：对幅值高斯滤波，相位不变 ==========
    print("\n实验①：对幅值进行高斯滤波（sigma=20）...")
    magnitude_filtered = apply_gaussian_to_magnitude(magnitude_original.copy(), sigma=20)
    result1 = reconstruct_from_components(magnitude_filtered, phase_original)
    
    # 显示滤波后的幅值谱
    magnitude_filtered_log = np.log(magnitude_filtered + 1)
    plt.subplot(2, 3, 4)
    plt.imshow(magnitude_filtered_log, cmap='gray')
    plt.title('4. Magnitude after Gaussian Filter')
    plt.axis('off')
    
    # 显示结果1
    plt.subplot(2, 3, 5)
    plt.imshow(result1, cmap='gray')
    plt.title('5. Result 1: Filter Magnitude + Original Phase')
    plt.axis('off')
    
    # ========== 实验②：幅值不变，对相位高斯滤波 ==========
    print("实验②：对相位进行高斯滤波（sigma=20）...")
    phase_filtered = apply_gaussian_to_phase(phase_original.copy(), sigma=20)
    result2 = reconstruct_from_components(magnitude_original, phase_filtered)
    
    # 显示滤波后的相位谱
    plt.subplot(2, 3, 6)
    plt.imshow(phase_filtered, cmap='gray')
    plt.title('6. Phase after Gaussian Filter')
    plt.axis('off')
    
    # 显示结果2（单独创建一个新的图形）
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(result1, cmap='gray')
    plt.title('Result 1: Filtered Magnitude + Original Phase')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result2, cmap='gray')
    plt.title('Result 2: Original Magnitude + Filtered Phase')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_2.3_results.png', dpi=150)
    plt.show()
    
    # 4. 输出分析结果
    print("\n" + "="*60)
    print("实验结果分析")
    print("="*60)
    
    print("\n【结果1：对幅值滤波 + 原相位】")
    print("- 图像整体变模糊（细节丢失）")
    print("- 但物体的轮廓和结构仍然清晰可辨")
    print("- 原因：相位谱保留了结构信息")
    
    print("\n【结果2：原幅值 + 对相位滤波】")
    print("- 图像结构严重破坏")
    print("- 几乎无法辨认物体")
    print("- 原因：相位信息被破坏，丢失了位置关系")
    
    print("\n【核心结论】")
    print("="*60)
    print("✓ 相位谱比幅值谱更重要！")
    print("✓ 相位信息决定了图像的结构和内容")
    print("✓ 幅值信息主要影响图像的纹理和细节")

def compare_sigma_effects():
    """比较不同sigma值对幅值和相位滤波的影响"""
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取 lena.jpg")
        return
    
    fshift = fourier_transform(img)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    sigmas = [5, 10, 20, 30]
    
    # 图1：对幅值滤波的结果
    fig1, axes1 = plt.subplots(1, len(sigmas) + 1, figsize=(15, 4))
    
    axes1[0].imshow(img, cmap='gray')
    axes1[0].set_title('Original')
    axes1[0].axis('off')
    
    for i, sigma in enumerate(sigmas):
        magnitude_filtered = apply_gaussian_to_magnitude(magnitude.copy(), sigma)
        result_mag = reconstruct_from_components(magnitude_filtered, phase)
        
        axes1[i + 1].imshow(result_mag, cmap='gray')
        axes1[i + 1].set_title(f'Filter Magnitude\nsigma={sigma}')
        axes1[i + 1].axis('off')
    
    fig1.suptitle('Effect of Magnitude Filtering (Phase fixed)', fontsize=14)
    plt.tight_layout()
    plt.savefig('hw_2.3_magnitude_filter.png', dpi=150)
    plt.show()
    
    # 图2：对相位滤波的结果
    fig2, axes2 = plt.subplots(1, len(sigmas) + 1, figsize=(15, 4))
    
    axes2[0].imshow(img, cmap='gray')
    axes2[0].set_title('Original')
    axes2[0].axis('off')
    
    for i, sigma in enumerate(sigmas):
        phase_filtered = apply_gaussian_to_phase(phase.copy(), sigma)
        result_phase = reconstruct_from_components(magnitude, phase_filtered)
        
        axes2[i + 1].imshow(result_phase, cmap='gray')
        axes2[i + 1].set_title(f'Filter Phase\nsigma={sigma}')
        axes2[i + 1].axis('off')
    
    fig2.suptitle('Effect of Phase Filtering (Magnitude fixed)', fontsize=14)
    plt.tight_layout()
    plt.savefig('hw_2.3_phase_filter.png', dpi=150)
    plt.show()
    
    print("\n【不同sigma值的影响总结】")
    print("-" * 50)
    print("对幅值滤波：sigma越大，图像越模糊，但轮廓保持")
    print("对相位滤波：即使sigma很小，也会严重破坏图像结构")
    print("结论：相位信息比幅值信息对图像重建更关键")

def compare_all_in_one():
    """在一个图中比较所有结果"""
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法读取 lena.jpg")
        return
    
    fshift = fourier_transform(img)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    sigmas = [5, 20, 40]
    
    fig, axes = plt.subplots(3, len(sigmas) + 1, figsize=(16, 10))
    
    # 第一行：原图 + 不同sigma下对幅值滤波的结果
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=12)
    axes[0, 0].axis('off')
    
    for i, sigma in enumerate(sigmas):
        magnitude_filtered = apply_gaussian_to_magnitude(magnitude.copy(), sigma)
        result_mag = reconstruct_from_components(magnitude_filtered, phase)
        axes[0, i + 1].imshow(result_mag, cmap='gray')
        axes[0, i + 1].set_title(f'Magnitude Filtered (σ={sigma})', fontsize=12)
        axes[0, i + 1].axis('off')
    
    # 第二行：不同sigma下对相位滤波的结果
    axes[1, 0].imshow(img, cmap='gray')
    axes[1, 0].set_title('Original', fontsize=12)
    axes[1, 0].axis('off')
    
    for i, sigma in enumerate(sigmas):
        phase_filtered = apply_gaussian_to_phase(phase.copy(), sigma)
        result_phase = reconstruct_from_components(magnitude, phase_filtered)
        axes[1, i + 1].imshow(result_phase, cmap='gray')
        axes[1, i + 1].set_title(f'Phase Filtered (σ={sigma})', fontsize=12)
        axes[1, i + 1].axis('off')
    
    # 第三行：频谱对比
    magnitude_log = np.log(magnitude + 1)
    axes[2, 0].imshow(magnitude_log, cmap='gray')
    axes[2, 0].set_title('Original Magnitude', fontsize=12)
    axes[2, 0].axis('off')
    
    for i, sigma in enumerate(sigmas):
        magnitude_filtered = apply_gaussian_to_magnitude(magnitude.copy(), sigma)
        magnitude_filtered_log = np.log(magnitude_filtered + 1)
        axes[2, i + 1].imshow(magnitude_filtered_log, cmap='gray')
        axes[2, i + 1].set_title(f'Filtered Magnitude (σ={sigma})', fontsize=12)
        axes[2, i + 1].axis('off')
    
    plt.suptitle('Experiment 2.3: Fourier Transform Analysis - Gaussian Filtering on Magnitude vs Phase', fontsize=14)
    plt.tight_layout()
    plt.savefig('hw_2.3_complete_comparison.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("实验2.3：傅里叶变换分析（滤波重构）")
    print("="*60)
    print("\n实验说明：")
    print("① 对幅值进行高斯滤波 + 原相位 → 重构图像")
    print("② 原幅值 + 对相位进行高斯滤波 → 重构图像")
    print("比较两种结果，理解幅值和相位的作用\n")
    
    print("请选择运行模式：")
    print("1. 基础实验（sigma=20的对比）")
    print("2. 不同sigma值对比（分开展示）")
    print("3. 综合对比（一张图展示所有结果）")
    
    choice = input("请输入选择 (1/2/3): ")
    
    if choice == "2":
        compare_sigma_effects()
    elif choice == "3":
        compare_all_in_one()
    else:
        main()