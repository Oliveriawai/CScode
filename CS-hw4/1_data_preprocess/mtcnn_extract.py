# mtcnn_extract.py
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import os
from pathlib import Path

def extract_faces_mtcnn(data_root, output_root, confidence_threshold=0.95, padding=0.05):
    """
    使用MTCNN提取人脸
    confidence_threshold: 置信度阈值（实验要求过滤低质量检测）
    padding: 边缘外扩比例（实验要求5%）
    """
    print(f"开始使用 MTCNN 处理数据集...")
    print(f"原始数据路径: {data_root}")
    print(f"输出路径: {output_root}")
    print(f"置信度阈值: {confidence_threshold}")
    print(f"边缘外扩: {padding*100}%")
    
    # 检查原始数据路径是否存在
    if not Path(data_root).exists():
        print(f"错误：原始数据路径不存在！请检查：{data_root}")
        return
    
    # 初始化MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    mtcnn = MTCNN(
        keep_all=True,          # 返回所有检测到的人脸
        device=device,
        post_process=False      # 不进行后处理，方便获取置信度
    )
    
    persons = [d for d in Path(data_root).iterdir() if d.is_dir()]
    
    if len(persons) == 0:
        print(f"错误：在 {data_root} 中没有找到任何文件夹！")
        return
    
    print(f"找到 {len(persons)} 个人的文件夹\n")
    
    # 统计信息
    total_images = 0
    detected_faces = 0
    filtered_faces = 0
    no_face_detected = 0
    
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
            
            try:
                # 读取图像
                img = Image.open(img_path).convert('RGB')
                
                # MTCNN检测（返回boxes和probs）
                boxes, probs = mtcnn.detect(img)
                
                if boxes is None or len(boxes) == 0:
                    no_face_detected += 1
                    print(f"未检测到人脸: {img_path.name}")
                    continue
                
                # 取置信度最高的检测结果
                best_idx = probs.argmax()
                confidence = probs[best_idx]
                box = boxes[best_idx]
                
                # 置信度过滤（实验要求）
                if confidence < confidence_threshold:
                    filtered_faces += 1
                    print(f"置信度过低({confidence:.3f} < {confidence_threshold}): {img_path.name}")
                    continue
                
                # 添加padding（5%外扩）
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                pad_w = int(w * padding)
                pad_h = int(h * padding)
                
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(img.size[0], x2 + pad_w)
                y2 = min(img.size[1], y2 + pad_h)
                
                # 裁剪并缩放到128x128
                face = img.crop((x1, y1, x2, y2))
                face_resized = face.resize((128, 128), Image.BILINEAR)
                
                # 保存（保持原格式或保存为jpg）
                output_path = Path(output_root) / person_name / f"{img_path.stem}_mtcnn.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                face_resized.save(output_path)
                
                detected_faces += 1
                
            except Exception as e:
                print(f"处理出错 {img_path.name}: {str(e)}")
                no_face_detected += 1
    
    # 打印统计结果
    print("\n" + "="*60)
    print("MTCNN 处理完成！统计结果：")
    print(f"总图片数: {total_images}")
    print(f"成功检测人脸: {detected_faces}")
    print(f"未检测到人脸: {no_face_detected}")
    print(f"置信度过低被过滤: {filtered_faces}")
    if total_images > 0:
        print(f"成功率: {detected_faces/total_images*100:.2f}%")
    print("="*60)
    
    # 保存统计信息到文件
    with open(Path(output_root) / "mtcnn_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"MTCNN 提取统计\n")
        f.write(f"{'='*40}\n")
        f.write(f"总图片数: {total_images}\n")
        f.write(f"成功检测人脸: {detected_faces}\n")
        f.write(f"未检测到人脸: {no_face_detected}\n")
        f.write(f"置信度过低被过滤: {filtered_faces}\n")
        f.write(f"成功率: {detected_faces/total_images*100:.2f}%\n")

if __name__ == "__main__":
    extract_faces_mtcnn(
        data_root="D:\\CS\\CScode\\CS-hw4\\CASIA-FaceV5",
        output_root="D:\\CS\\CScode\\CS-hw4\\extracted_faces_mtcnn",
        confidence_threshold=0.95,  # 实验要求的置信度阈值
        padding=0.05                 # 5%边缘外扩
    )