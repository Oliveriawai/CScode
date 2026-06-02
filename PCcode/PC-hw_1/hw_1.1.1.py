import cv2

img = cv2.imread('season.jpg')      # 读取图像
cv2.imshow('Window Title', img)     # 显示图像
cv2.waitKey(0)                      # 等待按键
cv2.destroyAllWindows()             # 关闭窗口