import numpy as np
import random


def softmax(z):
    z_exp = np.exp(z + 1e-6)
    partition = z_exp.sum(1, keepdims=True)
    return z_exp / partition


class MyMultiClassifier:

    # epochs 是最大训练轮数
    # rate 是学习率
    # e 是允许的误差
    # batch_size 是SDG的训练组的大小，默认使用全部即GD
    # alg 是采用的算法，默认是基础训练算法，可以选择 自适应学习率(adaptive), AdaGrad(adagrad), RMSprop(rmsprop), AdaDelta(adadelta)
    # 使用 AdaDetla 学习率不起作用
    # p 是RMSprop和AdaDelta中使用的衰减率, 默认是0.9
    # m 是动量因子, 默认是0, 通常是0.5
    # nesterov 默认False， 表示不开启 Nesterov 加速梯度
    def __init__(self, epochs, rate=0.01, e=1e-4, batch_size=0, alg="", p=0.9, m=0, nesterov=False):
        self.g = None
        self.theta = None
        self.epochs = epochs
        self.r = rate
        self.rate = self.r
        self.e = e
        self.t = 0
        self.list = []
        self.batch_size = batch_size
        self.sigma = None
        self.bce = np.inf
        self.ep = 1e-10
        self.p = p
        self.momentum = m
        self.lastV = 0
        self.nesterov = nesterov
        if nesterov:
            self.updateTheta = self.NAG
        if alg == "":
            self.fitTheta = self.simple
        elif alg == "adaptive":
            self.fitTheta = self.adaptive
        elif alg == "adagrad":
            self.fitTheta = self.adagrad
        elif alg == "rmsprop":
            self.fitTheta = self.RMSprop
        elif alg == "adadelta":
            self.diffXSquare = 0
            self.fitTheta = self.AdaDelta
        else:
            print("错误的算法: " + alg)
            self.fitTheta = self.simple

    def updateTheta(self, v):
        self.theta = self.theta - v

    def NAG(self, v):
        # 这里使用的是替换公式
        self.theta = self.theta - (1 + self.momentum) * v + self.momentum * self.lastV

    def simple(self, g):
        v = self.rate * g - self.momentum * self.lastV
        self.updateTheta(v)
        self.lastV = v

    def adaptive(self, g):
        v = self.rate / pow(self.t + 1, 0.5) * g - self.momentum * self.lastV
        self.updateTheta(v)
        self.lastV = v

    def adagrad(self, g):
        self.sigma += np.square(g)
        v = self.rate / (pow(self.sigma, 0.5) + self.ep) * g - self.momentum * self.lastV
        self.updateTheta(v)
        self.lastV = v

    def RMSprop(self, g):
        self.sigma = self.sigma * self.p + (1 - self.p) * np.square(g)
        v = self.rate / (pow(self.sigma + self.ep, 0.5)) * g - self.momentum * self.lastV
        self.updateTheta(v)
        self.lastV = v

    def AdaDelta(self, g):
        self.sigma = self.sigma * self.p + (1 - self.p) * np.square(g)
        v = pow(self.diffXSquare + self.ep, 0.5) / pow(self.sigma + self.ep, 0.5) * g - self.momentum * self.lastV
        self.updateTheta(v)
        self.lastV = v
        self.diffXSquare = self.p * self.diffXSquare + (1 - self.p) * v * v

    def getData(self, X, Y):
        if self.batch_size > 0:
            random.shuffle(self.list)
            batch_X = np.array([X[i] for i in range(self.batch_size)])
            batch_Y = np.array([Y[i] for i in range(self.batch_size)])
            return batch_X, batch_Y
        else:
            return X, Y

    def gradient(self, X, Y):
        z = X @ self.theta
        p = softmax(z)
        return -(X.T @ (Y - p)) / len(X)

    def fit(self, X, Y):
        self.t = 0
        self.list = [i for i in range(len(X))]
        Y = Y.reshape(len(Y), len(Y[0]))
        data = np.c_[np.ones(len(X)), X]
        self.rate = np.c_[np.ones(len(data[0]))] * self.r
        self.bce = np.inf
        self.theta = np.zeros((data.shape[1], Y.shape[1]))
        self.sigma = np.zeros((data.shape[1], Y.shape[1]))
        self.g = np.ones((data.shape[1], Y.shape[1]))
        while self.t < self.epochs and np.linalg.norm(self.g) > self.e:
            # 打乱索引
            batch_X, batch_Y = self.getData(data, Y)
            self.g = self.gradient(batch_X, batch_Y)
            self.fitTheta(self.g)
            self.t += 1
            self.bce = self.cross_entropy(X, Y)
        print("训练完成, BCE=" + str(self.bce) + " 训练" + str(self.t) + "次")

    def cross_entropy(self, X, Y):
        # 调整精度为1e-6
        data = np.c_[np.ones(len(X)), X]
        z = data @ self.theta
        y_hat = softmax(z)
        return - np.sum(Y * np.log(y_hat))

    def predict(self, X):
        data = np.c_[np.ones(len(X)), X]
        z = data @ self.theta
        p = softmax(z)
        y_pred = np.zeros((p.shape[0], p.shape[1]))
        for i in range(len(p)):
            y_pred[i][np.argmax(p[i])] = 1
        return y_pred
