# camera_recognition.py
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_database(db_path):
    """加载人脸数据库"""
    database = torch.load(db_path)
    print(f"加载数据库: {len(database)} 个人")
    return database

def main():
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化 MTCNN 和 FaceNet
    mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # 加载数据库
    database = load_database("step3_open_set/face_database.pt")
    
    # 识别阈值
    threshold = 0.80
    
    # 打开摄像头
    print("\n正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return
    
    print("✅ 摄像头已打开，按 'q' 退出，按 's' 截图保存")
    
    import time
    fps_counter = 0
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break
        
        # 转换 BGR 到 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # 检测人脸
        boxes, probs = mtcnn.detect(pil_img)
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.95:
                    continue
                
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # 提取人脸（返回 [C, H, W] 格式）
                face_tensor = mtcnn(pil_img)  # 返回 [N, C, H, W]
                
                if face_tensor is not None and i < len(face_tensor):
                    # 取第 i 个人脸，添加 batch 维度 [1, C, H, W]
                    single_face = face_tensor[i].unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        embedding = resnet(single_face).cpu().numpy()[0]
                    
                    # 与数据库匹配
                    best_sim = -1
                    best_name = None
                    for name, data in database.items():
                        if 'embeddings' in data:
                            sims = [cosine_similarity(embedding, emb) for emb in data['embeddings']]
                            sim = max(sims)
                        else:
                            sim = cosine_similarity(embedding, data['embedding'])
    
                        if sim > best_sim:
                            best_sim = sim
                            best_name = name
                    
                    # 判断是否认识
                    if best_sim >= threshold:
                        label = f"{best_name} ({best_sim:.2f})"
                        color = (0, 255, 0)
                    else:
                        label = f"Unknown ({best_sim:.2f})"
                        color = (0, 165, 255)
                    
                    # 绘制人脸框和标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 显示 FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            last_time = current_time
            cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示画面
        cv2.imshow('Face Recognition - Press q to quit', frame)
        
        # 按键控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{timestamp}.png", frame)
            print(f"截图已保存: screenshot_{timestamp}.png")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n摄像头已关闭")

if __name__ == "__main__":
    main()