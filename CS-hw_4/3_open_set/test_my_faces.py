# test_my_faces.py
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from pathlib import Path
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

database = torch.load("step3_open_set/face_database.pt")
print(f"数据库人数: {len(database)}")
print("="*50)

# 测试你的4张原始照片
test_dir = Path("my_faces")

for img_path in sorted(test_dir.glob("*.jpg")):
    print(f"\n照片: {img_path.name}")
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    
    if face is None:
        print(f"  ❌ 未检测到人脸")
        continue
    
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face).cpu().numpy()[0]
    
    # 找最相似的人
    best_sim = -1
    best_name = None
    for name, data in database.items():
        sim = cosine_similarity(embedding, data['embedding'])
        if sim > best_sim:
            best_sim = sim
            best_name = name
    
    threshold = 0.75
    if best_sim >= threshold:
        print(f"  ✅ 识别为: {best_name} (相似度: {best_sim:.4f})")
    else:
        print(f"  ❌ Unknown (最高相似度: {best_sim:.4f}, 来自: {best_name})")