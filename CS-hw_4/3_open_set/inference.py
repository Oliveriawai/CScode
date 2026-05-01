# 3_open_set/inference.py
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_database(db_path):
    """加载人脸数据库"""
    database = torch.load(db_path)
    print(f"加载数据库: {len(database)} 个人")
    return database

def recognize_face(image_path, database, mtcnn, resnet, device, threshold=0.75):
    """
    识别单张图片中的人脸
    返回: (person_name, similarity, face_box)
    """
    # 读取图片
    img = Image.open(image_path).convert('RGB')
    
    # 检测人脸
    boxes, probs = mtcnn.detect(img)
    
    if boxes is None or len(boxes) == 0:
        return None, None, None
    
    # 取置信度最高的人脸
    best_idx = np.argmax(probs)
    box = boxes[best_idx]
    
    # 提取人脸并获取特征
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None, None, box
    
    # 处理多个人脸的情况
    if len(face_tensor.shape) == 3:
        # 单个人脸 [C, H, W]
        single_face = face_tensor.unsqueeze(0).to(device)
    else:
        # 多个人脸 [N, C, H, W]
        if best_idx >= len(face_tensor):
            best_idx = 0
        single_face = face_tensor[best_idx].unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = resnet(single_face).cpu().numpy()[0]
    
    # 与数据库中所有人计算相似度（支持多张照片格式）
    best_sim = -1
    best_name = None
    
    for name, data in database.items():
        # 检查是新格式（多张照片）还是旧格式（单张）
        if 'embeddings' in data:
            # 多张照片：取最高相似度
            sims = [cosine_similarity(embedding, emb) for emb in data['embeddings']]
            sim = max(sims)
        else:
            # 单张照片（兼容旧格式）
            sim = cosine_similarity(embedding, data['embedding'])
        
        if sim > best_sim:
            best_sim = sim
            best_name = name
    
    # 判断是否认识（超过阈值）
    if best_sim >= threshold:
        return best_name, best_sim, box
    else:
        return "Unknown", best_sim, box

def visualize_result(img_path, result, similarity, box, save_path=None):
    """可视化识别结果"""
    img = plt.imread(img_path)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    
    # 绘制人脸框
    if box is not None:
        x1, y1, x2, y2 = box
        # 根据结果设置边框颜色
        if result == "Unknown":
            color = 'orange'
            label = f"Unknown ({similarity:.2f})"
        else:
            color = 'green'
            label = f"{result} ({similarity:.2f})"
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-10, label, fontsize=12, color=color, weight='bold')
    
    ax.axis('off')
    plt.title(f"识别结果: {result}")
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def test_casia_faces(database, mtcnn, resnet, device):
    """测试 CASIA 数据集中的前几个人"""
    print("\n" + "="*50)
    print("测试 CASIA 数据集")
    print("="*50)
    
    test_dir = Path("../data/extracted_faces_mtcnn")
    
    if not test_dir.exists():
        print(f"⚠️ 路径不存在: {test_dir}")
        return
    
    persons = list(test_dir.iterdir())[:5]
    
    for person in persons:
        img_files = list(person.glob('*.jpg'))
        if img_files:
            test_img = img_files[0]
            result, similarity, box = recognize_face(str(test_img), database, mtcnn, resnet, device)
            
            print(f"\n图片: {person.name}")
            print(f"  识别结果: {result}")
            print(f"  相似度: {similarity:.4f}")

def test_my_faces(database, mtcnn, resnet, device):
    """测试自己的多张照片"""
    print("\n" + "="*50)
    print("测试自己的照片")
    print("="*50)
    
    my_faces_dir = Path("../data/my_faces")
    
    if not my_faces_dir.exists():
        print(f"⚠️ 路径不存在: {my_faces_dir}")
        return
    
    for img_path in sorted(my_faces_dir.glob("*.jpg")):
        result, similarity, box = recognize_face(str(img_path), database, mtcnn, resnet, device)
        print(f"\n照片: {img_path.name}")
        print(f"  识别结果: {result}")
        print(f"  相似度: {similarity:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # 加载数据库（使用相对路径）
    database = load_database("face_database.pt")
    
    # 运行测试
    test_casia_faces(database, mtcnn, resnet, device)
    test_my_faces(database, mtcnn, resnet, device)

if __name__ == "__main__":
    main()