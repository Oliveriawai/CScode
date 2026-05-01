# align_my_faces.py
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
import torch

# 强制使用 CPU（避免 CUDA 兼容性问题）
device = torch.device('cpu')
print(f"使用设备: {device}")

# 初始化 MTCNN
mtcnn = MTCNN(image_size=160, margin=0, device=device)

input_dir = Path("my_faces")
output_dir = Path("my_faces_aligned")
output_dir.mkdir(exist_ok=True)

for img_path in sorted(input_dir.glob("*.jpg")):
    print(f"处理: {img_path.name}")
    img = Image.open(img_path).convert('RGB')
    
    # 获取对齐后的人脸
    face = mtcnn(img)
    
    if face is not None:
        # 转换为 PIL Image
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        face_pil = to_pil(face.cpu())
        
        output_path = output_dir / f"{img_path.stem}_aligned.jpg"
        face_pil.save(output_path)
        print(f"  ✅ 已保存: {output_path.name}")
    else:
        print(f"  ❌ 未检测到人脸")

print("\n完成！")