import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取图像
img = cv2.imread('season.jpg')

# 检查图像是否读取成功
if img is None:
    print("错误：无法读取图像，请确认 season.jpg 文件在当前目录下")
    exit()

# 2. 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 提取BGR三通道（OpenCV默认是BGR顺序）
b = img[:, :, 0]  # Blue通道
g = img[:, :, 1]  # Green通道
r = img[:, :, 2]  # Red通道

# 4. 转换为HSV颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:, :, 0]  # Hue色调 (0-179)
s = hsv[:, :, 1]  # Saturation饱和度 (0-255)
v = hsv[:, :, 2]  # Value明度 (0-255)

# 5. 使用matplotlib显示所有结果
plt.figure(figsize=(15, 12))

# 原图
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 灰度图
plt.subplot(2, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# RGB三通道
plt.subplot(2, 4, 3)
plt.imshow(r, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(g, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(b, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

# HSV三通道（全部显示！）
plt.subplot(2, 4, 6)
plt.imshow(h, cmap='hsv')
plt.title('Hue Channel')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(s, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(v, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

plt.tight_layout()
plt.savefig('hw_1.1.2_result.png', dpi=150)
plt.show()

# 也可以分别显示HSV三通道的OpenCV窗口
cv2.imshow('Hue', h)
cv2.imshow('Saturation', s)
cv2.imshow('Value', v)

print("结果已保存为 hw_1.2_result.png")
print(f"HSV范围 - Hue: {h.min()}-{h.max()}, Saturation: {s.min()}-{s.max()}, Value: {v.min()}-{v.max()}")
cv2.waitKey(0)
cv2.destroyAllWindows()