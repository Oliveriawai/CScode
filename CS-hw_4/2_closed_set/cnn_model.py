# step2_cnn/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """
    自定义CNN网络
    结构: 5层卷积 + BatchNorm + MaxPool + 3层全连接 + Dropout
    """
    def __init__(self, num_classes=100):
        super(CustomCNN, self).__init__()
        
        # 卷积层部分
        self.conv_layers = nn.Sequential(
            # 第1层: 输入 3x128x128, 输出 32x64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第2层: 输入 32x64x64, 输出 64x32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第3层: 输入 64x32x32, 输出 128x16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第4层: 输入 128x16x16, 输出 256x8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第5层: 输入 256x8x8, 输出 512x4x4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 计算全连接层的输入维度: 512 * 4 * 4 = 8192
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = CustomCNN(num_classes=100)
    print(f"模型总参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"输出形状: {out.shape}")