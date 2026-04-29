# add_new_person_multiview.py
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from pathlib import Path
import numpy as np

def add_multiple_views(person_name, image_paths, database_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    database = torch.load(database_path)
    
    embeddings = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = resnet(face).cpu().numpy()[0]
            embeddings.append(emb)
            print(f"  ✅ 已提取: {Path(img_path).name}")
    
    if embeddings:
        # 存储多个特征向量（列表形式）
        database[person_name] = {
            'embeddings': embeddings,  # 改为列表，支持多张
            'image_paths': image_paths
        }
        torch.save(database, database_path)
        print(f"✅ 成功添加 {person_name}，共 {len(embeddings)} 张照片")

if __name__ == "__main__":
    add_multiple_views(
        person_name="Oliveria",
        image_paths=[
            "my_faces/me_01.jpg",  # 正面
            "my_faces/me_02.jpg",  # 侧脸
            "my_faces/me_03.jpg",
            "my_faces/me_04.jpg"    
        ],
        database_path="step3_open_set/face_database.pt"
    )