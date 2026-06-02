import cv2
import numpy as np

def modify_neighborhood(img, x, y, size=16):
    """
    将图像中(x,y)坐标周围size×size邻域的像素值修改为(x,y)处的像素值
    """
    # 复制图像，避免修改原图
    img_copy = img.copy()
    
    # 获取图像尺寸
    h, w = img_copy.shape[:2]
    
    # 检查坐标是否有效
    if x < 0 or x >= w or y < 0 or y >= h:
        print(f"错误：坐标({x}, {y})超出图像范围({w}x{h})")
        return img_copy, None
    
    # 计算邻域边界
    half = size // 2
    x1 = max(0, x - half)
    x2 = min(w, x + half)
    y1 = max(0, y - half)
    y2 = min(h, y + half)
    
    # 获取中心点像素值
    center_pixel = img[y, x].copy()
    
    # 修改邻域内所有像素
    img_copy[y1:y2, x1:x2] = center_pixel
    
    # 在修改区域画一个红色边框，方便观察
    cv2.rectangle(img_copy, (x1, y1), (x2-1, y2-1), (0, 0, 255), 2)
    
    return img_copy, (x1, y1, x2, y2)

def main():
    # 1. 读取图像
    img = cv2.imread('lena.jpg')
    
    if img is None:
        print("错误：无法读取图像，请确认 lena.jpg 文件在当前目录下")
        return
    
    print(f"图像尺寸：{img.shape[1]} x {img.shape[0]}")
    print(f"图像类型：{img.shape[2]}通道彩色图像")
    
    # 2. 显示原图
    cv2.imshow('1. Original Image - Press any key to continue', img)
    cv2.waitKey(0)  # 等待按键后继续
    
    # 3. 获取用户输入的坐标
    print("\n" + "="*50)
    print("请输入要修改的中心点坐标：")
    
    try:
        x = int(input(f"请输入 x 坐标（0-{img.shape[1]-1}）："))
        y = int(input(f"请输入 y 坐标（0-{img.shape[0]-1}）："))
    except ValueError:
        print("输入无效，使用默认坐标 (200, 200)")
        x, y = 200, 200
    
    # 4. 修改邻域像素
    modified_img, region = modify_neighborhood(img, x, y, size=16)
    
    if region:
        x1, y1, x2, y2 = region
        print(f"\n修改区域：x=[{x1}, {x2}), y=[{y1}, {y2})")
        print(f"中心点({x}, {y})的像素值(BGR)：{img[y, x]}")
    
    # 5. 显示修改后的图像
    cv2.imshow('2. Modified Image - Press any key to save and exit', modified_img)
    
    # 6. 保存结果
    cv2.imwrite('hw_1.3_result.png', modified_img)
    print("\n结果已保存为 hw_1.3_result.png")
    
    # 7. 等待按键后关闭所有窗口
    print("\n按任意键关闭所有窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()