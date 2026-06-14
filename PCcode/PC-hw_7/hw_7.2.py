import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_blur_laplacian(image_path):
    """
    使用拉普拉斯方差检测图像模糊
    这是最常用、最可靠的方法
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算拉普拉斯方差
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance, gray

def main():
    image_dir = 'blur-det'
    images = ['adrian_01.png', 'adrian_02.png', 'jemma.png', 'resume.png']
    
    results = []
    
    print("="*60)
    print("图像模糊检测结果（基于拉普拉斯方差）")
    print("="*60)
    print(f"{'文件名':<20} {'拉普拉斯方差':<14} {'判断结果':<10}")
    print("-"*60)
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"文件不存在：{img_path}")
            continue
        
        variance, gray = detect_blur_laplacian(img_path)
        
        # 阈值设为100：方差小于100判定为模糊
        # 根据你的数据：jemma.png=73.77 < 100，其他三张都大于100
        is_blur = variance < 100
        
        status = "模糊" if is_blur else "清晰"
        results.append((img_name, variance, is_blur))
        
        # 用颜色标记结果
        if is_blur:
            print(f"{img_name:<20} {variance:<14.2f} {status:<10}  ← 检测为模糊")
        else:
            print(f"{img_name:<20} {variance:<14.2f} {status:<10}")
    
    print("="*60)
    
    # 统计
    blur_count = sum(1 for r in results if r[2])
    clear_count = len(results) - blur_count
    print(f"\n统计：模糊 {blur_count} 张，清晰 {clear_count} 张")
    
    # 可视化显示
    visualize_results(results, image_dir)

def visualize_results(results, image_dir):
    """可视化显示检测结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (img_name, variance, is_blur) in enumerate(results):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        color = 'red' if is_blur else 'green'
        title = f"{img_name}\n{'模糊' if is_blur else '清晰'}\n方差={variance:.2f}"
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(title, color=color, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_7_2_result.png', dpi=150)
    plt.show()

def analyze_threshold():
    """分析阈值设置"""
    image_dir = 'blur-det'
    images = ['adrian_01.png', 'adrian_02.png', 'jemma.png', 'resume.png']
    
    print("\n" + "="*60)
    print("拉普拉斯方差分析")
    print("="*60)
    
    scores = []
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue
        
        variance, _ = detect_blur_laplacian(img_path)
        scores.append(variance)
        print(f"{img_name}: 方差 = {variance:.2f}")
    
    print("-"*60)
    print(f"最小值: {min(scores):.2f}")
    print(f"最大值: {max(scores):.2f}")
    print(f"平均值: {np.mean(scores):.2f}")
    
    print("\n" + "="*60)
    print("阈值建议")
    print("="*60)
    print("从数据看：")
    print("  - jemma.png 方差=73.77，明显偏低")
    print("  - 其他三张方差都在1000以上")
    print("\n建议阈值设为 100~500 之间")
    print("当前代码使用阈值：100")
    print("  - 方差 < 100 → 模糊")
    print("  - 方差 ≥ 100 → 清晰")

if __name__ == "__main__":
    print("实验7-2：图像模糊检测")
    print("="*50)
    print("1. 检测四张图片")
    print("2. 分析阈值设置")
    
    choice = input("请选择 (1/2): ")
    
    if choice == "2":
        analyze_threshold()
    else:
        main()