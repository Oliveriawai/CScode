# haar_extract.py
import cv2
import os
from pathlib import Path

def detect_face_haar(image_path, cascade, output_path, padding=0.05):
    """
    使用Haar级联检测人脸并裁剪
    padding: 边缘外扩比例（实验要求5%）
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    
    if len(faces) == 0:
        return False
    
    # 取第一个检测到的人脸
    x, y, w, h = faces[0]
    
    # 添加padding（5%外扩）
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(img.shape[1] - x, w + 2 * pad_w)
    h = min(img.shape[0] - y, h + 2 * pad_h)
    
    # 裁剪并缩放到128x128
    face = img[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (128, 128))
    
    cv2.imwrite(output_path, face_resized)
    return True

def process_dataset_haar(data_root, output_root):
    """处理整个CASIA-FaceV5数据集"""
    print(f"开始处理数据集...")
    print(f"原始数据路径: {data_root}")
    print(f"输出路径: {output_root}")
    
    # 检查原始数据路径是否存在
    if not Path(data_root).exists():
        print(f"错误：原始数据路径不存在！请检查：{data_root}")
        return
    
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    persons = [d for d in Path(data_root).iterdir() if d.is_dir()]
    
    if len(persons) == 0:
        print(f"错误：在 {data_root} 中没有找到任何文件夹！")
        print("请确认文件夹结构是否为：CASIA-FaceV5/人员编号/照片.jpg")
        return
    
    print(f"找到 {len(persons)} 个人的文件夹\n")
    
    # 统计信息
    total_images = 0
    detected_faces = 0
    failed_images = 0
    
    for idx, person_dir in enumerate(persons, 1):
        person_name = person_dir.name
        # 支持 .bmp, .jpg, .png, .jpeg 格式
        img_files = (list(person_dir.glob('*.bmp')) + 
                    list(person_dir.glob('*.jpg')) + 
                    list(person_dir.glob('*.png')) + 
                    list(person_dir.glob('*.jpeg')))
        
        print(f"[{idx}/{len(persons)}] 正在处理: {person_name} (共 {len(img_files)} 张图片)")
        
        for img_path in img_files:
            total_images += 1
            output_path = Path(output_root) / person_name / f"{img_path.stem}_haar.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if detect_face_haar(str(img_path), cascade, str(output_path)):
                detected_faces += 1
            else:
                failed_images += 1
                print(f"未检测到人脸: {img_path.name}")
    
    # 打印统计结果
    print("\n" + "="*50)
    print("处理完成！统计结果：")
    print(f"总图片数: {total_images}")
    print(f"成功检测人脸: {detected_faces}")
    print(f"失败（未检测到人脸）: {failed_images}")
    if total_images > 0:
        print(f"成功率: {detected_faces/total_images*100:.2f}%")
    else:
        print("警告：没有找到任何图片文件！")
    print("="*50)

if __name__ == "__main__":
    process_dataset_haar("D:\\CS\\CScode\\CS-hw4\\CASIA-FaceV5", "D:\\CS\\CScode\\CS-hw4\\extracted_faces_haar")