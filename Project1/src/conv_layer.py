import numpy as np
from src.linear_layer import LinearLayer
import src.utils as u


class ConvLayer(LinearLayer):
    # 卷积层，继承全连接层的功能，支持前向传播、反向传播和权重更新
    def __init__(self, input_size, k, f=3, s=1, p=1, u_type='adam', a_type='relu', dropout=1):
        """
            卷积层
            参数（其他两层类似）：
                input_size: 输入数据的形状 (C, H, W)，其中 C 是通道数，H 是高度，W 是宽度
                k: 卷积核的数量（输出通道数）
                f: 卷积核的大小，默认为 3
                s: 步长，默认为 1
                p: padding,默认为 1
                u_type: 更新算法类型，默认为 'adam'
                a_type: 激活函数类型，默认为 'relu'
                dropout: Dropout 比率，默认为 1（即不使用 dropout）
            """
        self.image_size = 0
        self.w = input_size[2]
        self.h = input_size[1]
        self.d = input_size[0]

        self.k = k
        self.f = f
        self.s = s
        self.p = p

        self.new_w = int((self.w - self.f + 2 * self.p) / self.s + 1)
        self.new_h = int((self.h - self.f + 2 * self.p) / self.s + 1)
        self.new_d = k

        super(ConvLayer, self).__init__(f*f*self.d, k, u_type=u_type, a_type=a_type, dropout=dropout)

    def predict(self, batch):
        '''
        用于预测的前向传播
        '''
        self.image_size = batch.shape[0]
        cols = u.im2col_indices(batch, self.f, self.f, self.p, self.s)
        sum_weights = []
        bias = []
        for n in self.neurons:
            bias.append(n.bias)
            sum_weights.append(n.weights)

        sum_weights = np.array(sum_weights)
        strength = (sum_weights.dot(cols) + np.array(bias).reshape(sum_weights.shape[0], 1)).reshape(self.k, self.new_h, self.new_w, -1).transpose(3, 0, 1, 2)

        if self.activation:
            if self.a_type == 'sigmoid':
                return u.sigmoid(strength)
            else:
                return u.relu(strength)
        else:
            return strength

    def forward(self, batch):
        '''
        用于训练的前向传播
        :param batch: (N,C,H,W)形状
        :return: 输出和正则化项
        '''
        self.image_size = batch.shape[0]
        cols = u.im2col_indices(batch, self.f, self.f, self.p, self.s)
        l2 = 0
        sum_weights = []
        bias = []
        for n in self.neurons:
            n.last_input = cols
            sum_weights.append(n.weights)
            bias.append(n.bias)
            l2 += n.regularization()

        sum_weights = np.array(sum_weights)
        strength = (sum_weights.dot(cols) + np.array(bias).reshape(sum_weights.shape[0], 1))
        strength = strength.reshape(self.k, self.new_h, self.new_w, -1).transpose(3, 0, 1, 2)

        if self.activation:
            if self.a_type == 'sigmoid':
                self.forward_result = u.sigmoid(strength)
            else:
                self.forward_result = u.relu(strength)
        else:
            self.forward_result = strength

        return self.forward_result, l2

    def backward(self, gradiant, need_gradiant=True):
        '''
        反向传播（计算梯度）
        :param gradiant: loss function对当前层输出的梯度
        :param need_gradiant: 是否需要梯度数据
        :return: 输入数据的梯度
        '''
        if gradiant.ndim < 4:
            gradiant = gradiant.reshape(self.new_w, self.new_h, self.k, -1).T

        if self.activation:
            if self.a_type == 'sigmoid':
                delta = gradiant * u.sigmoid_d(self.forward_result)
            else:
                delta = gradiant * u.relu_d(self.forward_result)
        else:
            delta = gradiant

        sum_weights = []
        for index, n in enumerate(self.neurons):
            n.grad_z = delta[:, index, :, :].transpose(1, 2, 0).flatten()
            if need_gradiant:
                rot = np.rot90(n.weights.reshape(self.d, self.f, self.f), k=2, axes=(1, 2))
                sum_weights.append(rot)

        if not need_gradiant:
            return

        padding = ((self.w - 1) * self.s + self.f - self.new_w) // 2
        cols = u.im2col_indices(delta, self.f, self.f, padding=padding, stride=self.s)

        sum_weights = np.array(sum_weights).transpose(1, 0, 2, 3).reshape(self.d, -1)

        result = sum_weights.dot(cols)
        current_gradiant = result.reshape(self.d, self.h, self.w, -1).transpose(3, 0, 1, 2)

        return current_gradiant

    def output_size(self):
        return (self.new_d, self.new_h, self.new_w)

    def update(self, lr, l2_reg, t=0):
        super(ConvLayer, self).update(lr, l2_reg, t)
