from src.linear_layer import LinearLayer
from src.conv_layer import ConvLayer
from src.pool_layer import PoolLayer
import src.utils as utils
import numpy as np

class NeuralNetwork(object):
    #支持卷积层、池化层、全连接层，并提供前向传播、反向传播和权重更新功能
    def __init__(self, input_shape, layer_list, lr, l2_reg=0, loss='softmax'):
        self.layers = []
        self.lr = np.float32(lr)
        self.l2_reg = np.float32(l2_reg)
        self.loss = loss

        self.dropout_masks = [] #存储 dropout 掩码
        self.t = 0

        next_layer_input = input_shape
        for layer in layer_list:
            if layer['type'] == 'conv':
                layer.pop('type')
                conv = ConvLayer(next_layer_input, **layer)
                self.layers.append(conv)
                next_layer_input = conv.output_size()

            elif layer['type'] == 'pool':
                layer.pop('type')
                pool = PoolLayer(next_layer_input, **layer)
                self.layers.append(pool)
                next_layer_input = pool.output_size()

            elif layer['type'] == 'fc':
                layer.pop('type')
                fc = LinearLayer(next_layer_input, **layer)
                self.layers.append(fc)
                next_layer_input = fc.output_size()

            elif layer['type'] == 'output':
                layer.pop('type')
                fc = LinearLayer(next_layer_input, **layer)
                fc.is_output = True
                fc.activation = False
                self.layers.append(fc)
                next_layer_input = fc.output_size()

    def predict(self, batch, label):
        next_input = batch
        for index, layer in enumerate(self.layers):
            next_input = layer.predict(next_input)

        result = np.array(next_input)
        if self.loss == 'softmax':
            loss, delta = utils.softmax_loss(result, label)
        elif self.loss == 'logistic':
            loss, delta = utils.logistic_loss(result, label)

        max_result = np.argmax(result, axis=0)
        correct_count = np.sum(max_result == label)

        return loss, correct_count / float(len(max_result)) * 100

    def epoch(self, batch, label):
        # 前向传播
        l2 = 0
        next_input = batch
        for index, layer in enumerate(self.layers):
            layer_result = layer.forward(next_input)
            next_input = layer_result[0]
            l2 += layer_result[1]
            if layer.dropout < 1 and not layer.is_output:
                dropout_mask = np.random.rand(*next_input.shape) < layer.dropout
                next_input *= dropout_mask / layer.dropout
                self.dropout_masks.append(dropout_mask)


        result = np.array(next_input)
        if self.loss == 'softmax':
            loss, gradiant = utils.softmax_loss(result, label)
        elif self.loss == 'logistic':
            loss, gradiant = utils.logistic_loss(result, label)

        loss += 0.5 * self.l2_reg * l2
        max_result = np.argmax(result, axis=0)
        correct_count = np.sum(max_result == label)

        # 反向传播
        back_input = gradiant.T
        for index, layer in enumerate(reversed(self.layers)):
            is_input_layer = index < len(self.layers) - 1

            if layer.dropout < 1 and not layer.is_output and self.dropout_masks:
                dropout_mask = self.dropout_masks.pop()
                if dropout_mask.ndim > 2 and back_input.ndim == 2:
                    back_input *= dropout_mask.T.reshape(-1, back_input.shape[1])
                else:
                    back_input *= dropout_mask

            back_input = layer.backward(back_input, is_input_layer)

        # 动态更新训练参数
        for index, layer in enumerate(self.layers):
            layer.update(self.lr, l2_reg=self.l2_reg, t=self.t)

        return loss + self.l2_reg * l2 / 2, correct_count / float(len(max_result)) * 100
