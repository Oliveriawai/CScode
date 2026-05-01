# test_resnet18.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_model(num_classes=100, model_path='best_resnet18_model.pth'):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def predict_image(model, device, image_path, class_to_idx):
    """预测单张图片"""
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = probabilities[0][predicted].item()
    
    # 将索引转回类别名
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class[predicted.item()]
    
    return predicted_class, confidence

def main():
    # 需要先运行训练脚本，保存 class_to_idx
    # 临时方案：从 data_loader 获取
    from data_loader import get_data_loaders
    
    data_root = "D:\\CS\\CScode\\CS-hw4\\extracted_faces_mtcnn"
    _, _, class_to_idx = get_data_loaders(data_root, batch_size=8)
    
    # 加载模型
    model, device = load_model(num_classes=len(class_to_idx))
    print(f"模型加载成功，使用设备: {device}")
    
    # 选择测试图片（从验证集中选几个）
    test_dir = Path(data_root)
    persons = list(test_dir.iterdir())
    
    # 测试5张不同人的图片
    for person in persons[:5]:
        images = list(person.glob('*.jpg'))
        if images:
            test_img = images[0]
            pred_class, confidence = predict_image(model, device, str(test_img), class_to_idx)
            
            # 显示结果
            img = plt.imread(test_img)
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title(f"真实: {person.name}\n预测: {pred_class}\n置信度: {confidence:.2f}")
            plt.axis('off')
            plt.show()
            
            print(f"真实: {person.name}, 预测: {pred_class}, 置信度: {confidence:.2f}")

if __name__ == "__main__":
    main()