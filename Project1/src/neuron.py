import numpy as np


class Neuron(object):
    def __init__(self, input_size, bias=0.0):
        '''初始化'''
        self.weights = (np.random.randn(input_size) * np.sqrt(2.0 / input_size)).astype(np.float32)
        #用 He Normal 初始化 生成权重 → 为 ReLU 准备, 防止前向传播时激活值过大或过小
        self.bias = np.float32(bias)
        self.last_input = None # 上一次输入，用于反向传播
        self.grad_z = None #Loss对z的偏导，用于权重更新

    def strength(self, values):
        '''计算输出值（线性变换）'''
        return np.dot(self.weights, values) + self.bias

    def regularization(self):
        '''采用L2正则化'''
        return np.sum(np.square(self.weights))