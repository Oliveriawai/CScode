# step3_open_set/inference.py
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
    confidence = probs[best_idx]
    
    # 提取人脸并获取特征
    face = mtcnn(img)
    if face is None:
        return None, None, box
    
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face).cpu().numpy()[0]
    
    # 与数据库中所有人计算相似度
    best_sim = -1
    best_name = None
    
    for name, data in database.items():
        db_embedding = data['embedding']
        sim = cosine_similarity(embedding, db_embedding)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    
    # 判断是否认识（超过阈值）
    if best_sim >= threshold:
        return best_name, best_sim, box
    else:
        return "Unknown", best_sim, box

def visualize_result(img_path, result, box, save_path=None):
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
            label = f"{result} (相似度: {box[0]:.2f})" if isinstance(box, tuple) else result
        else:
            color = 'green'
            label = f"{result}"
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-10, label, fontsize=12, color=color, weight='bold')
    
    ax.axis('off')
    plt.title(f"识别结果: {result}")
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # 加载数据库
    database = load_database("D:\\CS\\CScode\\CS-hw4\\step3_open_set\\face_database.pt")
    
    # 测试图片路径（可以修改成你想要测试的图片）
    test_dir = Path("D:\\CS\\CScode\\CS-hw4\\extracted_faces_mtcnn")
    
    # 测试5个人
    persons = list(test_dir.iterdir())[:5]
    
    for person in persons:
        img_files = list(person.glob('*.jpg'))
        if img_files:
            test_img = img_files[0]
            result, similarity, box = recognize_face(str(test_img), database, mtcnn, resnet, device)
            
            print(f"\n图片: {person.name}")
            print(f"识别结果: {result}")
            if similarity:
                print(f"相似度: {similarity:.4f}")
            
            # 可选：显示图片
            # visualize_result(str(test_img), result, box)
            def test_my_faces():
                """测试自己的多张照片"""
                my_faces_dir = Path("D:\\CS\\CScode\\CS-hw4\\my_faces")
    
                for img_path in my_faces_dir.glob("*.jpg"):
                    result, similarity, box = recognize_face(str(img_path), database, mtcnn, resnet, device)
                    print(f"照片: {img_path.name}")
                    print(f"  识别结果: {result}, 相似度: {similarity:.4f}")

if __name__ == "__main__":
    main()