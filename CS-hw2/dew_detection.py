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

# 2. 转HSV，提取亮白区域
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
lower_dew = np.array([0, 0, 190])
upper_dew = np.array([180, 60, 255])
mask = cv2.inRange(hsv, lower_dew, upper_dew)

# 3. 形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

# 4. 霍夫圆检测
max_radius = min(width, height) // 20
min_radius = 3

circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=50, param2=30,
                           minRadius=min_radius, maxRadius=max_radius)

dew_count = 0
if circles is not None:
    circles = np.uint16(np.around(circles))
    dew_count = len(circles[0, :])
    
    for i in circles[0, :]:
        if i[2] > max_radius or i[2] < min_radius:
            continue
        cv2.circle(output, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(output, (i[0], i[1]), 2, (0, 255, 0), 3)

# 5. 计算露珠百分比
total_pixels = mask.size
dew_pixels = cv2.countNonZero(mask)
dew_percentage = (dew_pixels / total_pixels) * 100

print(f"检测出露珠数量: {dew_count}")
print(f"检测出露珠占图片的百分比: {dew_percentage:.2f}%")

# 保存检测结果
cv2.imwrite(r"dew_detected.png", output)
print("结果已保存到 dew_detected.png")

# 不显示窗口，直接退出
print("程序运行完成！")