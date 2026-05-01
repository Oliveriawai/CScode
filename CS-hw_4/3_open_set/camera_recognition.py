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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # ========== 修改这里 ==========
    database = load_database("face_database.pt")  # 改成当前目录下的文件
    # =============================
    
    threshold = 0.75
    
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
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        boxes, probs = mtcnn.detect(pil_img)
        
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.95:
                    continue
                
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face_tensor = mtcnn(pil_img)
                
                if face_tensor is not None and i < len(face_tensor):
                    single_face = face_tensor[i].unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        embedding = resnet(single_face).cpu().numpy()[0]
                    
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
                    
                    if best_sim >= threshold:
                        label = f"{best_name} ({best_sim:.2f})"
                        color = (0, 255, 0)
                    else:
                        label = f"Unknown ({best_sim:.2f})"
                        color = (0, 165, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        fps_counter += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            last_time = current_time
            cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition - Press q to quit', frame)
        
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