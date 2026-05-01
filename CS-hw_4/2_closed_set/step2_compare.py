# step2_compare.py
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(cnn_history, resnet_history):
    """对比CNN和ResNet的训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 训练损失对比
    axes[0,0].plot(cnn_history['train_loss'], label='Custom CNN', linewidth=2)
    axes[0,0].plot(resnet_history['train_loss'], label='ResNet18', linewidth=2)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 验证损失对比
    axes[0,1].plot(cnn_history['val_loss'], label='Custom CNN', linewidth=2)
    axes[0,1].plot(resnet_history['val_loss'], label='ResNet18', linewidth=2)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].set_title('Validation Loss Comparison')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 训练准确率对比
    axes[1,0].plot(cnn_history['train_acc'], label='Custom CNN', linewidth=2)
    axes[1,0].plot(resnet_history['train_acc'], label='ResNet18', linewidth=2)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy (%)')
    axes[1,0].set_title('Training Accuracy Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 验证准确率对比
    axes[1,1].plot(cnn_history['val_acc'], label='Custom CNN', linewidth=2)
    axes[1,1].plot(resnet_history['val_acc'], label='ResNet18', linewidth=2)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].set_title('Validation Accuracy Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('cnn_vs_resnet_comparison.png')
    plt.show()
    
    # 打印最终对比
    print("\n" + "="*60)
    print("最终结果对比")
    print("="*60)
    print(f"Custom CNN - 最佳验证准确率: {max(cnn_history['val_acc']):.2f}%")
    print(f"ResNet18 - 最佳验证准确率: {max(resnet_history['val_acc']):.2f}%")
    print(f"CNN vs ResNet: {max(resnet_history['val_acc']) - max(cnn_history['val_acc']):.2f}% 提升")

if __name__ == "__main__":
    # 这里需要先运行两个训练脚本，保存history到文件
    print("请先运行 step2_cnn/train_cnn.py 和 step2_resnet/train_resnet.py")
    print("然后将训练历史保存下来，再运行此对比脚本")