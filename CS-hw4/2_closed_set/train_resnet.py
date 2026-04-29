# step2_resnet/train_resnet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import get_data_loaders

def create_resnet18_model(num_classes, freeze_layers=True):
    """
    创建ResNet18模型
    freeze_layers: 是否冻结前部卷积层（只微调深层）
    """
    model = models.resnet18(pretrained=True)
    
    if freeze_layers:
        # 冻结前3个layer（layer1, layer2, layer3）
        for name, param in model.named_parameters():
            if 'layer1' in name or 'layer2' in name or 'layer3' in name:
                param.requires_grad = False
            # layer4和fc层保持可训练
            elif 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            # 其他层（如bn层）也冻结
            else:
                param.requires_grad = False
        
        print("冻结策略: 只训练 layer4 和 fc 层")
    else:
        # 全部训练
        print("冻结策略: 全部层都训练")
    
    # 替换全连接层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def count_trainable_parameters(model):
    """计算可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    # 配置
    data_root = "D:\\CS\\CScode\\CS-hw4\\extracted_faces_mtcnn"
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    freeze_layers = True  # 是否冻结前部卷积层
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print("="*60)
    
    # 加载数据
    train_loader, val_loader, class_to_idx = get_data_loaders(data_root, batch_size)
    num_classes = len(class_to_idx)
    print(f"类别数: {num_classes}")
    
    # 创建模型
    model = create_resnet18_model(num_classes, freeze_layers=freeze_layers).to(device)
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 损失函数和优化器（只优化可训练的参数）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("\n开始训练...")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        # 调整学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch [{epoch}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet18_model.pth')
            print(f'  ★ 保存最佳模型 (Acc: {val_acc:.2f}%)')
        
        print("-"*60)
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    # 绘制训练曲线
    plot_training_curves(history, 'resnet18_training_curves.png')
    
    return history

def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('ResNet18: Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('ResNet18: Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练曲线已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    history = main()