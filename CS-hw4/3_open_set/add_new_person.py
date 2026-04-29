# add_new_person.py
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from pathlib import Path
import numpy as np

def add_multiple_views(person_name, image_paths, database_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    database = torch.load(database_path)
    print(f"当前数据库人数: {len(database)}")
    
    embeddings = []
    valid_paths = []
    
    for img_path in image_paths:
        # 确保路径是 Path 对象
        img_path = Path(img_path)
        print(f"处理: {img_path}")
        
        # 检查文件是否存在
        if not img_path.exists():
            print(f"  ❌ 文件不存在: {img_path}")
            continue
            
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = resnet(face).cpu().numpy()[0]
            embeddings.append(emb)
            valid_paths.append(str(img_path))
            print(f"  ✅ 特征提取成功")
        else:
            print(f"  ❌ 未检测到人脸")
    
    if embeddings:
        database[person_name] = {
            'embeddings': embeddings,
            'image_paths': valid_paths
        }
        torch.save(database, database_path)
        print(f"\n✅ 成功添加 {person_name}，共 {len(embeddings)} 张照片")
        print(f"当前数据库人数: {len(database)}")
    else:
        print("❌ 没有有效的人脸照片")

if __name__ == "__main__":
    add_multiple_views(
        person_name="Oliveria",
        image_paths=[
            "../data/my_faces/me_01.jpg",
            "../data/my_faces/me_02.jpg",
            "../data/my_faces/me_03.jpg",
            "../data/my_faces/me_04.jpg",
        ],
        database_path="face_database.pt"
    )