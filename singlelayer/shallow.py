# encoding=utf-8
__author__ = 'hujintao'
from layers import *


def main():
    datalayer1 = Data('data\\train.npy', 1024)  # 用于训练，batch_size设置为1024
    datalayer2 = Data('data\\validate.npy', 10000)  # 用于验证，所以设置batch_size为10000,一次性计算所有的样例
    datalayer3 = Data('data\\test.npy', 10000)
    inner_layers = []
    inner_layers.append(FullyConnect(17 * 17, 26))
    inner_layers.append(Sigmoid())
    losslayer = QuadraticLoss()
    accuracy = Accuracy()

    for layer in inner_layers:
        layer.lr = 1000.0  # 为所有中间层设置学习速率

    epochs = 20
    for i in range(epochs):
        print 'epochs:', i
        losssum = 0
        iters = 0
        while True:
            data, pos = datalayer1.forward()  # 从数据层取出数据
            x, label = data
            for layer in inner_layers:  # 前向计算
                x = layer.forward(x)

            loss = losslayer.forward(x, label)  # 调用损失层forward函数计算损失函数值
            losssum += loss
            iters += 1
            d = losslayer.backward()  # 调用损失层backward函数层计算将要反向传播的梯度

            for layer in inner_layers[::-1]:  # 反向传播
                d = layer.backward(d)

            if pos == 0:  # 一个epoch完成后进行准确率测试
                data, _ = datalayer2.forward()
                x, label = data
                for layer in inner_layers:
                    x = layer.forward(x)
                accu = accuracy.forward(x, label)  # 调用准确率层forward()函数求出准确率
                print 'loss:', losssum / iters
                print 'accuracy:', accu
                break
    data, _ = datalayer3.forward()
    x, label = data
    for layer in inner_layers:
        x = layer.forward(x)

    xxx = np.array([np.argmax(xx) for xx, ll in zip(x, label)])
    print "预测结果如下："
    print xxx[0:20]


if __name__ == '__main__':
    main()