import cv2
import numpy as np

# 读取原图
image = cv2.imread(r"dew.jpg")
if image is None:
    print("无法读取图片，请检查文件路径")
    exit()

output = image.copy()
height, width = image.shape[:2]

# 1. 双边滤波 + 高斯模糊 去噪
blurred = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

# 2. 转HSV，提取亮白区域（放宽参数）
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
lower_dew = np.array([0, 0, 150])      # V下限从190降到150，检测更多区域
upper_dew = np.array([180, 70, 255])  # S上限从70升到100，允许更多颜色变化
mask = cv2.inRange(hsv, lower_dew, upper_dew)

# 3. 形态学操作（减少过滤，保留小露珠）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)    # 开运算从2次减到1次
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)   # 闭运算保持1次

# 可选：去除太小的连通域（降低最小面积要求）
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 2  # 最小面积从5降到3，保留更小的露珠
new_mask = np.zeros_like(mask)
for contour in contours:
    area = cv2.contourArea(contour)
    if area >= min_area:
        cv2.drawContours(new_mask, [contour], -1, 255, -1)
mask = new_mask

# 4. 霍夫圆检测（降低阈值，检测更多圆）
max_radius = min(width, height) // 35  # 最大半径放宽
min_radius = 2                          # 最小半径从3降到2

circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,  # minDist从60降到40
                           param1=50, param2=28,  # param1从50降到40，param2从40降到30
                           minRadius=min_radius, maxRadius=max_radius)

# 5. 验证和计数露珠（放宽填充比例要求）
dew_count = 0
valid_circles = []

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    for i in circles[0, :]:
        if i[2] > max_radius or i[2] < min_radius:
            continue
        
        # 验证圆内露珠像素填充比例（放宽条件）
        center_x, center_y, radius = i[0], i[1], i[2]
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)
        
        circle_area = np.pi * radius * radius
        dew_in_circle = cv2.bitwise_and(mask, mask, mask=circle_mask)
        dew_pixels_in_circle = cv2.countNonZero(dew_in_circle)
        fill_ratio = dew_pixels_in_circle / circle_area if circle_area > 0 else 0
        
        # 放宽填充比例范围（0.1-0.98），原来0.2-0.95
        if 0.1 < fill_ratio < 0.98:
            valid_circles.append(i)
            cv2.circle(output, (center_x, center_y), radius, (0, 0, 255), 2)
            cv2.circle(output, (center_x, center_y), 2, (0, 255, 0), 3)
    
    dew_count = len(valid_circles)

# 6. 计算露珠百分比
total_pixels = mask.size
dew_pixels = cv2.countNonZero(mask)
dew_percentage = (dew_pixels / total_pixels) * 100

print(f"检测出露珠数量: {dew_count}")
print(f"检测出露珠占图片的百分比: {dew_percentage:.2f}%")

# 保存检测结果
cv2.imwrite(r"dew_detected.png", output)
print("结果已保存到 dew_detected.png")

# 保存mask用于调试
cv2.imwrite(r"mask_debug.png", mask)
print("mask已保存到 mask_debug.png，可查看检测区域")

print("程序运行完成！")