import sys, os
sys.path.insert(1, os.path.split(os.path.split(sys.path[0])[0])[0])

import pickle as pkl
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from src.neural_net import NeuralNetwork
from tools import save_model, plot_curves, visualize_conv_rgb_layer1, visualize_conv_rgb_layer2, visualize_conv_rgb_layer3
import copy

# ---------- 数据处理 ----------
def unpickle(file):
    with open(file, 'rb') as fo:
        return pkl.load(fo, encoding='latin1')

le = preprocessing.LabelEncoder()
le.classes_ = unpickle(sys.path[0] + '/batches.meta')['label_names']

# 加载训练数据
train_images, train_labels = None, []
for i in range(1, 6):
    data = unpickle(sys.path[0] + f'/data_batch_{i}')
    train_images = data['data'] if train_images is None else np.vstack((train_images, data['data']))
    train_labels += data['labels']

train_images = train_images.reshape(-1, 3, 32, 32).astype(np.float64)
mean_image = np.mean(train_images, axis=0)
std = np.std(train_images, axis=0)
train_images = ((train_images - mean_image) / std).astype(np.float32)

# 验证集划分
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
val_images = train_images[:4000]
val_labels = train_labels[:4000]
train_images = train_images[4000:]
train_labels = train_labels[4000:]

# 测试集处理
test_data = unpickle(sys.path[0] + '/test_batch')
test_images = test_data['data'].reshape(-1, 3, 32, 32).astype(np.float64)
test_images = ((test_images - mean_image) / std).astype(np.float32)
test_labels = test_data['labels']

# ---------- 模型超参数 ----------
lr = 1e-4
l2_reg = 8e-6
decay = 0.96
batch_size = 1
num_epochs = 20 #之后loss开始回升

cnn = NeuralNetwork(
    train_images.shape[1:],
    [
        {'type': 'conv', 'k': 16, 'u_type': 'nag', 'f': 5, 's': 1, 'p': 2},
        {'type': 'pool', 'method': 'average'},
        {'type': 'conv', 'k': 20, 'u_type': 'nag', 'f': 5, 's': 1, 'p': 2},
        {'type': 'pool', 'method': 'average'},
        {'type': 'conv', 'k': 20, 'u_type': 'nag', 'f': 5, 's': 1, 'p': 2},
        {'type': 'pool', 'method': 'average'},
        {'type': 'output', 'k': len(le.classes_), 'u_type': 'adam'}
    ],
    lr, l2_reg=l2_reg
)

# ---------- 训练主循环 ----------
best_model = None
best_val_acc = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    # Shuffle 每个 epoch
    train_images, train_labels = shuffle(train_images, train_labels)

    # mini-batch 训练
    for i in range(0, len(train_images), batch_size):
        x_batch = train_images[i:i+batch_size]
        y_batch = train_labels[i:i+batch_size]
        cnn.t += 1
        loss, acc = cnn.epoch(x_batch, y_batch)

    # epoch 完成：评估
    train_loss, train_acc = cnn.predict(train_images[:4000], train_labels[:4000])
    val_loss, val_acc = cnn.predict(val_images, val_labels)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - lr={cnn.lr:.6f}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(cnn)
        print(f"--> New best model saved (val acc = {val_acc:.4f})")

    # 衰减学习率
    cnn.lr *= decay

# ---------- 测试集评估 ----------
#test_loss, test_acc = best_model.predict(test_images, test_labels)
#print(f"\nFinal Test Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")

# ---------- 保存模型 ----------
#save_model(best_model, "best_model.pkl")
#print(f"\nBest model saved as 'best_model.pkl' with val acc = {best_val_acc:.4f}")

# ---------- 可视化 ----------
#plot_curves(train_losses, val_losses, train_accs, val_accs)

#print("First Layer Conv Kernels:")
#visualize_conv_rgb_layer1(best_model.layers[0], 16)

#print("Second Layer Conv Kernels:")
#visualize_conv_rgb_layer2(best_model.layers[2], 20)

#print("Third Layer Conv Kernels:")
#visualize_conv_rgb_layer3(best_model.layers[4], 20)
