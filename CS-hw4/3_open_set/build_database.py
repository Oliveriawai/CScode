# step3_open_set/build_database.py
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from pathlib import Path
import pickle
import numpy as np

def build_face_database(data_root, output_path='face_database.pt'):
    """
    构建人脸特征数据库
    为数据集中的每个人提取一张照片的512维特征向量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化 MTCNN 人脸检测和 InceptionResnetV1 特征提取
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    data_root = Path(data_root)
    persons = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    database = {}
    
    print(f"找到 {len(persons)} 个人，开始提取特征...")
    
    for idx, person_dir in enumerate(persons):
        person_name = person_dir.name
        
        # 取该人员的第一张照片作为"证件照"
        img_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
        if not img_files:
            print(f"⚠️ {person_name}: 没有找到图片")
            continue
        
        img_path = img_files[0]
        
        # 读取并检测人脸
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        
        if face is None:
            print(f"⚠️ {person_name}: 人脸检测失败")
            continue
        
        # 提取特征向量 (512维)
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face).cpu().numpy()[0]
        
        database[person_name] = {
            'embedding': embedding,
            'image_path': str(img_path)
        }
        
        print(f"[{idx+1}/{len(persons)}] {person_name}: 特征提取完成")
    
    # 保存数据库
    torch.save(database, output_path)
    print(f"\n✅ 数据库已保存至: {output_path}")
    print(f"共 {len(database)} 个人")
    return database

if __name__ == "__main__":
    # 使用 MTCNN 提取的人脸数据
    build_face_database(
        data_root="../data/extracted_faces_mtcnn",  # ✅ 新路径（相对路径）
        output_path="face_database.pt"  # ✅ 新路径（保存在当前文件夹）
)