import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

#-----------损失函数-------------------
def softmax_loss(x, y):
    """
        计算 Softmax损失函数及其梯度
        输入：
            x: 模型的输出，形状为 (N, C)，其中 N 是样本数量，C 是类别数量
            y: 真实标签，形状为 (N,)，每个值是 [0, C-1] 范围内的整数
        输出：
            loss: Softmax 损失值
            dx: 损失函数对输入 x 的梯度
    """
    x = x.T
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[range(N), y])) / N
    dx = probs
    dx[range(N), y] -= 1
    dx /= N
    return loss, dx


def logistic_loss(x, y):
    """
        计算 Logistic损失函数及其梯度
        输入：
            x: 模型的输出，形状为 (N, C)
            y: 真实标签，形状为 (N,)
        输出：
            loss: Logistic 损失值
            dx: 损失函数对输入 x 的梯度
    """
    N = x.shape[0]
    loss = np.sum(np.square(y - x) / 2) / N
    dx = -(y - x)
    return loss, dx.T

#-------------卷积层相关操作--------------------
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
        获取将输入数据转换为列（im2col）操作所需的索引，方便计算。
        输入：
            x_shape: 输入数据的形状 (N, C, H, W)
            field_height: 卷积核高度
            field_width: 卷积核宽度
            padding: 填充大小，默认为 1
            stride: 步长，默认为 1
        输出：
            k: 通道索引
            i: 高度索引
            j: 宽度索引
    """
    N, C, H, W = x_shape
    # 检查输出尺寸为整数
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    # 输出大小
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    # 计算索引
    i_0 = np.repeat(np.arange(field_height), field_width)
    i_0 = np.tile(i_0, C)
    i_1 = stride * np.repeat(np.arange(out_height), out_width)
    j_0 = np.tile(np.arange(field_width), field_height * C)
    j_1 = stride * np.tile(np.arange(out_width), out_height)
    i = i_0.reshape(-1, 1) + i_1.reshape(1, -1)
    j = j_0.reshape(-1, 1) + j_1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
        将输入数据转换为列（im2col）
        输入：
            x: 输入数据，形状为 (N, C, H, W)
            field_height: 卷积核高度
            field_width: 卷积核宽度
            padding: 填充大小，默认为 1
            stride: 步长，默认为 1
        输出：
            cols: 转换后的列矩阵，形状为 (field_height * field_width * C, out_height * out_width * N)
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
        将列矩阵转换回原始输入数据形状（col2im）。
        输入：
            cols: 列矩阵，形状为 (field_height * field_width * C, out_height * out_width * N)
            x_shape: 原始输入数据的形状 (N, C, H, W)
            field_height: 卷积核高度
            field_width: 卷积核宽度
            padding: 填充大小，默认为 1
            stride: 步长，默认为 1
        输出：
            x_padded: 转换后的数据，形状为 (N, C, H + 2*padding, W + 2*padding)。
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

#-----------更新神经元权重-----------------------
def rmsprop(neurons, lr, l2_reg=0, decay_rate=0.9, eps=1e-8):
    """
        使用 RMSProp 优化算法更新神经元权重
        输入：
            neurons: 神经元
            lr: 学习率
            l2_reg: L2 正则化系数
            decay_rate: 衰减率
            eps: 防止除零的小扰动
        """
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.grad_z)).T + l2
        d_bias = np.sum(n.grad_z)

        n.cache = decay_rate * n.cache + (1 - decay_rate) * (dx ** 2)
        n.weights += - lr * dx / (np.sqrt(n.cache) + eps)
        n.bias -= lr * d_bias

def adam_update(neurons, lr, t, l2_reg=0, beta1=np.float32(0.9), beta2=np.float32(0.999), eps=1e-8):
    """
        使用 Adam 优化算法更新神经元权重
        输入：
            neurons: 神经元
            lr: 学习率
            t: 当前迭代次数
            l2_reg: L2 正则化系数
            beta1: 一阶矩估计的衰减率
            beta2: 二阶矩估计的衰减率
            eps: 防止除零的小扰动
        """
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.grad_z)).T + l2
        d_bias = np.sum(n.grad_z)

        n.m = beta1 * n.m + (1 - beta1) * dx
        n.v = beta2 * n.v + (1 - beta2) * (dx**2)

        m = n.m / np.float32(1-beta1**t)
        v = n.v / np.float32(1-beta2**t)

        n.weights -= lr * m / (np.sqrt(v) + eps)
        n.bias -= lr * d_bias


def nag_update(neurons, lr, l2_reg=0, mu=np.float32(0.9)):
    """
        使用 Nesterov Accelerated Gradient (NAG) 优化算法更新神经元权重
        输入：
            neurons: 神经元
            lr: 学习率
            l2_reg: L2 正则化系数
            mu: 动量系数
    """
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.grad_z)).T + l2
        d_bias = np.sum(n.grad_z)

        n.v_prev = n.v
        n.v = mu * n.v - lr * dx

        n.weights += -mu * n.v_prev + (1 + mu) * n.v
        n.bias -= lr * d_bias


def momentum_update(neurons, lr, l2_reg=0, mu=np.float32(0.9)):
    """
        使用 Momentum 优化算法更新神经元权重
        输入：
            neurons: 神经元
            lr: 学习率
            l2_reg: L2 正则化系数
            mu: 动量系数
    """
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.grad_z)).T + l2
        d_bias = np.sum(n.grad_z)

        # 更新速度（考虑动量和梯度）
        n.v = mu * n.v - lr * dx
        n.weights += n.v

        # 更新偏置的速度
        n.v_bias = mu * n.v_bias - lr * d_bias
        n.bias += n.v_bias


def vanila_update(neurons, lr, l2_reg=0):
    """
        使用 Vanilla Gradient Descent 优化算法更新神经元权重
        输入：
            neurons: 神经元
            lr: 学习率
            l2_reg: L2 正则化系数
    """
    for n in neurons:
        l2 = l2_reg * n.weights
        dx = (n.last_input.dot(n.grad_z)).T + l2
        d_bias = np.sum(n.grad_z)

        n.weights -= lr * dx + l2
        n.bias -= lr * d_bias

#---------激活函数及其导数-----------------
def sigmoid(input):
    return 1/(1+np.exp(-input))


def relu(input):
    return np.maximum(0, input)


def sigmoid_d(input):
    return input * (1 - input)


def relu_d(input):
    return input > 0


