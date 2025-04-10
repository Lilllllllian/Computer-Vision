import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def save_model(model, filename):
    """保存模型为 pickle文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


def load_model(filename):
    """从 pickle文件加载模型"""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练,验证的 Loss和 Accuracy曲线"""

    epochs = np.arange(1, len(train_losses) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # -------- Loss 图 --------
    axs[0].plot(epochs, train_losses, label='Train Loss', color='tab:blue')
    axs[0].plot(epochs, val_losses, label='Validation Loss', color='tab:orange')
    axs[0].set_title("Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # -------- Accuracy 图 --------
    axs[1].plot(epochs, train_accs, label='Train Accuracy', color='tab:green')
    axs[1].plot(epochs, val_accs, label='Validation Accuracy', color='tab:red')
    axs[1].set_title("Accuracy Curve")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def visualize_conv_rgb_layer1(conv_layer, num_kernels=16):
    '''可视化第一层卷积层'''
    fig, axes = plt.subplots(1, num_kernels, figsize=(num_kernels * 1.5, 2))
    for i in range(num_kernels):
        if i >= len(conv_layer.neurons): break
        weights_1d = conv_layer.neurons[i].weights
        kernel = weights_1d.reshape(3, 5, 5).transpose(1, 2, 0)
        kernel -= kernel.min()
        kernel /= kernel.max() + 1e-8
        axes[i].imshow(kernel)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def visualize_conv_rgb_layer2(conv_layer, num_kernels=20):
    '''可视化第二层卷积层'''
    fig, axes = plt.subplots(1, num_kernels, figsize=(num_kernels * 1.5, 2))
    for i in range(num_kernels):
        if i >= len(conv_layer.neurons): break
        weights_1d = conv_layer.neurons[i].weights
        kernel = weights_1d.reshape(16, 5, 5)  #input channel:16
        kernel = kernel[:3].transpose(1, 2, 0) # 取前3个通道作为 RGB 可视化
        kernel -= kernel.min()
        kernel /= kernel.max() + 1e-8
        axes[i].imshow(kernel)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def visualize_conv_rgb_layer3(conv_layer, num_kernels=20):
    '''可视化第三层卷积层'''
    fig, axes = plt.subplots(1, num_kernels, figsize=(num_kernels * 1.5, 2))
    for i in range(num_kernels):
        if i >= len(conv_layer.neurons): break
        weights_1d = conv_layer.neurons[i].weights
        kernel = weights_1d.reshape(20, 5, 5)  # input channel:20
        kernel = kernel[:3].transpose(1, 2, 0)
        kernel -= kernel.min()
        kernel /= kernel.max() + 1e-8
        axes[i].imshow(kernel)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

