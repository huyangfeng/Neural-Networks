# encoding=utf-8
__author__ = 'hujintao'
import sys
from scipy import misc
import numpy as np


def main():
    # l = len(sys.argv)
    # if l < 2:
    #     print'eg: python img2pkl.py list.txt dst.npy\n' \
    #          'convert image to npy\n'
    #     return
    #
    # src = sys.argv[1]
    # dst = sys.argv[2] if l > 2 else 'data.pkl'

    src = "data\\test.txt"
    dst = "data\\test.npy"
    with open(src, 'r') as f:  # 读取图片列表
        list = f.readlines()

    data = []
    labels = []
    for i in list:
        name, label = i.strip('\n').split(' ')  # 将图片列表中的每一行拆分成图片名和图片标签
        print name + ' processed'
        img = misc.imread("data/" +name)  # 将图片读取出来，存入一个矩阵
        img /= 255  # 将图片转换为只有0、1值的矩阵
        img.resize((img.size, 1))  # 为了之后的运算方便，我们将图片存储到一个img.size*1的列向量里面
        data.append(img)
        labels.append(int(label))
        print data
        print labels


    print 'write to npy'
    np.save(dst, [data, labels])  # 将训练数据以npy的形式保存到成本地文件
    print 'completed'


if __name__ == '__main__':
    main()