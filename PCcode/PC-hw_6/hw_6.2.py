import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 读取图像
    img = cv2.imread('left01.jpg')
    
    if img is None:
        print("错误：找不到 left01.jpg")
        print("请将图像文件放在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. SIFT特征点检测
    sift = cv2.SIFT_create()
    keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
    
    # 3. ORB特征点检测
    orb = cv2.ORB_create(nfeatures=500)
    keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
    
    # 4. 绘制特征点
    img_sift = cv2.drawKeypoints(img_rgb, keypoints_sift, None, 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_orb = cv2.drawKeypoints(img_rgb, keypoints_orb, None, 
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 5. 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(img_sift)
    plt.title(f'SIFT: {len(keypoints_sift)} keypoints')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_orb)
    plt.title(f'ORB: {len(keypoints_orb)} keypoints')
    plt.axis('off')
    
    # 特征点数量对比柱状图
    plt.subplot(2, 2, 4)
    methods = ['SIFT', 'ORB']
    counts = [len(keypoints_sift), len(keypoints_orb)]
    colors = ['blue', 'orange']
    plt.bar(methods, counts, color=colors, alpha=0.7)
    plt.title('Keypoints Comparison')
    plt.ylabel('Number of Keypoints')
    for i, v in enumerate(counts):
        plt.text(i, v + 5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('hw_6.2_result.png', dpi=150)
    plt.show()
    
    # 6. 输出分析
    print("\n" + "="*50)
    print("特征点检测结果分析")
    print("="*50)
    print(f"SIFT特征点数量：{len(keypoints_sift)}")
    print(f"ORB特征点数量：{len(keypoints_orb)}")
    
    # 计算特征点平均响应强度
    sift_responses = [kp.response for kp in keypoints_sift]
    orb_responses = [kp.response for kp in keypoints_orb]
    
    print(f"\nSIFT平均响应强度：{np.mean(sift_responses):.2f}")
    print(f"ORB平均响应强度：{np.mean(orb_responses):.2f}")
    
    # 计算特征点分布（图像中心区域比例）
    h, w = gray.shape
    center_h, center_w = h//2, w//2
    
    def center_ratio(keypoints, size=100):
        center_count = 0
        for kp in keypoints:
            x, y = kp.pt
            if abs(x - center_w) < size and abs(y - center_h) < size:
                center_count += 1
        return center_count / len(keypoints) if len(keypoints) > 0 else 0
    
    print(f"\nSIFT中心区域占比：{center_ratio(keypoints_sift)*100:.1f}%")
    print(f"ORB中心区域占比：{center_ratio(keypoints_orb)*100:.1f}%")

def compare_different_images():
    """在不同图像上比较SIFT和ORB"""
    images = ['left01.jpg', 'harmadik2.jpg']
    titles = ['left01.jpg (Checkerboard)', 'harmadik2.jpg (Building)']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (img_name, title) in enumerate(zip(images, titles)):
        img = cv2.imread(img_name)
        if img is None:
            print(f"警告：无法读取 {img_name}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # SIFT
        sift = cv2.SIFT_create()
        kp_sift, _ = sift.detectAndCompute(gray, None)
        img_sift = cv2.drawKeypoints(img_rgb, kp_sift, None, 
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # ORB
        orb = cv2.ORB_create(nfeatures=500)
        kp_orb, _ = orb.detectAndCompute(gray, None)
        img_orb = cv2.drawKeypoints(img_rgb, kp_orb, None, 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f'{title}\nOriginal')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img_sift)
        axes[idx, 1].set_title(f'SIFT: {len(kp_sift)} points')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(img_orb)
        axes[idx, 2].set_title(f'ORB: {len(kp_orb)} points')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.2_comparison.png', dpi=150)
    plt.show()

def compare_orb_parameters():
    """比较ORB算子的不同参数"""
    img = cv2.imread('left01.jpg')
    
    if img is None:
        print("错误：找不到 left01.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 不同参数配置
    configs = [
        (100, "nfeatures=100"),
        (300, "nfeatures=300"),
        (500, "nfeatures=500"),
        (1000, "nfeatures=1000"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    for i, (nfeatures, title) in enumerate(configs):
        orb = cv2.ORB_create(nfeatures=nfeatures)
        kp, _ = orb.detectAndCompute(gray, None)
        img_kp = cv2.drawKeypoints(img_rgb, kp, None, color=(0, 255, 0))
        
        row = (i + 1) // 3
        col = (i + 1) % 3
        axes[row, col].imshow(img_kp)
        axes[row, col].set_title(f'ORB: {title}\n({len(kp)} points)')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.2_orb_params.png', dpi=150)
    plt.show()
    
    print("\nORB参数影响：")
    print("nfeatures越大，检测到的特征点越多")
    print("但过多的特征点会增加计算量，且可能包含重复或无效点")

def compare_sift_contrast_threshold():
    """比较SIFT算子的不同对比度阈值"""
    img = cv2.imread('left01.jpg')
    
    if img is None:
        print("错误：找不到 left01.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 不同对比度阈值
    thresholds = [0.01, 0.03, 0.05, 0.1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    for i, thresh in enumerate(thresholds):
        sift = cv2.SIFT_create(contrastThreshold=thresh)
        kp, _ = sift.detectAndCompute(gray, None)
        img_kp = cv2.drawKeypoints(img_rgb, kp, None, 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        row = (i + 1) // 3
        col = (i + 1) % 3
        axes[row, col].imshow(img_kp)
        axes[row, col].set_title(f'SIFT: contrast={thresh}\n({len(kp)} points)')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.2_sift_params.png', dpi=150)
    plt.show()
    
    print("\nSIFT对比度阈值影响：")
    print("阈值越低，检测到的特征点越多（包括弱特征）")
    print("阈值越高，只保留强特征点")

def feature_matching_demo():
    """特征点匹配演示（使用两张图）"""
    img1 = cv2.imread('left01.jpg')
    img2 = cv2.imread('left02.jpg') if cv2.imread('left02.jpg') is not None else img1
    
    if img1 is None:
        print("错误：找不到 left01.jpg")
        return
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # SIFT匹配
    sift = cv2.SIFT_create()
    kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
    kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)
    
    # FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches_sift = flann.knnMatch(des1_sift, des2_sift, k=2)
    
    # 筛选好的匹配点
    good_matches_sift = []
    for m, n in matches_sift:
        if m.distance < 0.7 * n.distance:
            good_matches_sift.append(m)
    
    img_match_sift = cv2.drawMatches(img1, kp1_sift, img2, kp2_sift, 
                                      good_matches_sift[:50], None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_match_sift, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Matching: {len(good_matches_sift)} matches')
    plt.axis('off')
    
    # ORB匹配（如果第二张图存在）
    if img2 is not None:
        orb = cv2.ORB_create(nfeatures=500)
        kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
        kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_orb = bf.match(des1_orb, des2_orb)
        matches_orb = sorted(matches_orb, key=lambda x: x.distance)[:50]
        
        img_match_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, 
                                         matches_orb, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_match_orb, cv2.COLOR_BGR2RGB))
        plt.title(f'ORB Matching: {len(matches_orb)} matches')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('hw_6.2_matching.png', dpi=150)
    plt.show()
    
    print(f"\nSIFT匹配点数量：{len(good_matches_sift)}")

def analyze_keypoint_properties():
    """分析特征点属性（位置、大小、响应强度）"""
    img = cv2.imread('left01.jpg')
    
    if img is None:
        print("错误：找不到 left01.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp_sift, _ = sift.detectAndCompute(gray, None)
    
    orb = cv2.ORB_create(nfeatures=500)
    kp_orb, _ = orb.detectAndCompute(gray, None)
    
    # 提取属性
    def extract_properties(keypoints):
        sizes = [kp.size for kp in keypoints]
        responses = [kp.response for kp in keypoints]
        angles = [kp.angle for kp in keypoints if kp.angle != -1]
        return sizes, responses, angles
    
    sizes_sift, resp_sift, angles_sift = extract_properties(kp_sift)
    sizes_orb, resp_orb, angles_orb = extract_properties(kp_orb)
    
    # 绘制分布图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].hist(sizes_sift, bins=30, color='blue', alpha=0.7)
    axes[0, 0].set_title('SIFT Keypoint Size Distribution')
    axes[0, 0].set_xlabel('Size')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(resp_sift, bins=30, color='blue', alpha=0.7)
    axes[0, 1].set_title('SIFT Response Distribution')
    axes[0, 1].set_xlabel('Response')
    
    axes[0, 2].hist(angles_sift, bins=36, color='blue', alpha=0.7)
    axes[0, 2].set_title('SIFT Orientation Distribution')
    axes[0, 2].set_xlabel('Angle (degrees)')
    
    axes[1, 0].hist(sizes_orb, bins=30, color='orange', alpha=0.7)
    axes[1, 0].set_title('ORB Keypoint Size Distribution')
    axes[1, 0].set_xlabel('Size')
    axes[1, 0].set_ylabel('Count')
    
    axes[1, 1].hist(resp_orb, bins=30, color='orange', alpha=0.7)
    axes[1, 1].set_title('ORB Response Distribution')
    axes[1, 1].set_xlabel('Response')
    
    axes[1, 2].hist(angles_orb, bins=36, color='orange', alpha=0.7)
    axes[1, 2].set_title('ORB Orientation Distribution')
    axes[1, 2].set_xlabel('Angle (degrees)')
    
    plt.tight_layout()
    plt.savefig('hw_6.2_properties.png', dpi=150)
    plt.show()
    
    print("\n特征点属性统计：")
    print(f"SIFT - 平均大小：{np.mean(sizes_sift):.2f}, 平均响应：{np.mean(resp_sift):.2f}")
    print(f"ORB  - 平均大小：{np.mean(sizes_orb):.2f}, 平均响应：{np.mean(resp_orb):.2f}")

if __name__ == "__main__":
    print("实验6.2：图像特征点检测")
    print("="*50)
    print("1. 基础实验（SIFT vs ORB）")
    print("2. 不同图像对比")
    print("3. ORB参数对比（nfeatures）")
    print("4. SIFT参数对比（contrastThreshold）")
    print("5. 特征点匹配演示")
    print("6. 特征点属性分析")
    
    choice = input("请选择 (1/2/3/4/5/6): ")
    
    if choice == "2":
        compare_different_images()
    elif choice == "3":
        compare_orb_parameters()
    elif choice == "4":
        compare_sift_contrast_threshold()
    elif choice == "5":
        feature_matching_demo()
    elif choice == "6":
        analyze_keypoint_properties()
    else:
        main()