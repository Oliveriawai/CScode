# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path

class FaceDataset(Dataset):
    """人脸数据集加载器"""
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # 获取所有人员文件夹
        persons = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        # 构建类别映射
        for idx, person_dir in enumerate(persons):
            self.class_to_idx[person_dir.name] = idx
            
            # 读取该人员所有图片
            img_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')) + list(person_dir.glob('*.bmp'))
            for img_path in img_files:
                self.images.append(img_path)
                self.labels.append(idx)
        
        print(f"数据集加载完成: {len(self.images)} 张图片, {len(persons)} 个类别")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 读取图片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_root, batch_size=8, train_ratio=0.8):
    """获取训练集和验证集的DataLoader"""
    
    # 数据增强（训练集）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 验证集（只做基本预处理）
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载完整数据集
    full_dataset = FaceDataset(data_root, transform=train_transform)
    
    # 划分训练集和验证集
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 为验证集单独设置transform
    val_dataset.dataset.transform = val_transform
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    
    return train_loader, val_loader, full_dataset.class_to_idx