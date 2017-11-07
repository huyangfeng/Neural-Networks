# encoding=utf-8
__author__ = 'hujintao'
import numpy as np


class Data:
    def __init__(self, name, batch_size):  # 数据所在的文件名name和batch中图片的数量batch_size
        with open(name, 'rb') as f:
            data = np.load(f)
        self.x = data[0]  # 输入x data的第一行
        self.y = data[1]  # 预期正确输出y
        self.l = len(self.x)
        self.batch_size = batch_size
        self.pos = 0  # pos用来记录数据读取的位置

    def forward(self):
        pos = self.pos
        bat = self.batch_size
        l = self.l
        if pos + bat >= l:  # 已经是最后一个batch时，返回剩余的数据，并设置pos为开始位置0
            ret = (self.x[pos:l], self.y[pos:l])
            self.pos = 0
            index = range(l)
            np.random.shuffle(index)  # 将训练数据打乱
            self.x = self.x[index]
            self.y = self.y[index]
        else:  # 不是最后一个batch, pos直接加上batch_size
            ret = (self.x[pos:pos + bat], self.y[pos:pos + bat])
            self.pos += self.batch_size

        return ret, self.pos  # 返回的pos为0时代表一个epoch已经结束

    def backward(self, d):  # 数据层无backward操作
        pass


class FullyConnect:
    def __init__(self, l_x, l_y):  # 两个参数分别为输入层的长度和输出层的长度
        self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x)  # 使用随机数初始化参数，请暂时忽略这里为什么多了np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)  # 使用随机数初始化参数
        self.lr = 0  # 先将学习速率初始化为0，最后统一设置学习速率

    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])  # 计算全连接层的输出
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d):
        ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.db = np.sum(d, axis=0) / self.x.shape[0]
        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])

        # 更新参数
        self.weights -= self.lr * self.dw
        self.bias -= self.lr * self.db
        return self.dx  # 反向传播梯度


class Sigmoid:
    def __init__(self):  # 无参数，不需初始化
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1 - sig)
        return self.dx  # 反向传递梯度


class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)  # 由于我们的label本身只包含一个数字，我们需要将其转换成和模型输出值尺寸相匹配的向量形式
        for a, b in zip(self.label, label):
            a[b] = 1.0  # 只有正确标签所代表的位置概率为1，其他为0
        self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2  # 求平均后再除以2是为了表示方便
        return self.loss

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]  # 2被抵消掉了
        return self.dx


class Accuracy:
    def __init__(self):
        pass

    def forward(self, x, label):  # 只需forward
        self.accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])  # 对预测正确的实例数求和
        self.accuracy = 1.0 * self.accuracy / x.shape[0]
        return self.accuracy