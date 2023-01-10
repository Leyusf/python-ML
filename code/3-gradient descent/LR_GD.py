import numpy as np
import random


class LR_GD:

    # epochs 是最大训练轮数
    # rate 是学习率
    # e 是允许的误差
    def __init__(self, epochs, rate=0.01, e=1e-7, alr=False, batch_size=0, ada=False):
        self.theta = None
        self.epochs = epochs
        self.rate = rate
        self.e = e
        self.alr = alr
        self.t = 0
        self.list = []
        self.batch_size = batch_size
        self.ada = ada
        self.g = None
        self.mse = np.inf

    def Rate(self, d):
        # 自适应学习率
        rate = np.c_[np.ones(d)] * self.rate
        if self.alr:
            if self.ada:
                return rate / pow(self.g * self.g, 0.5)
            return rate / pow(self.t + 1, 0.5)
        return rate

    def getData(self, X, Y):
        if self.batch_size > 0:
            random.shuffle(self.list)
            batch_X = np.array([X[i] for i in range(self.batch_size)])
            batch_Y = np.array([Y[i] for i in range(self.batch_size)])
            return batch_X, batch_Y
        else:
            return X, Y

    def gradient(self, X, Y):
        # y = b0 + b1x1 + b2x2 + ... + bnxn
        return -2 / len(X) * X.T @ (Y - X @ self.theta)

    def fit(self, X, Y):
        self.t = 0
        self.list = [i for i in range(len(X))]
        Y = Y.reshape(len(Y), 1)
        data = np.c_[np.ones(len(X)), X]
        self.mse = np.inf
        self.theta = np.c_[np.zeros(len(data[0]))]
        self.g = np.c_[np.zeros(len(data[0]))]
        size = len(self.theta)
        while self.t < self.epochs and self.mse > self.e:
            # 打乱索引
            batch_X, batch_Y = self.getData(data, Y)
            g = self.gradient(batch_X, batch_Y)
            self.g += g
            v = self.Rate(size) * g
            self.theta = self.theta - v
            self.t += 1
            self.mse = np.ones(len(batch_Y)) @ ((batch_Y - self._predict(batch_X)) * (batch_Y - self._predict(batch_X)))
        print("训练完成, MSE=" + str(self.mse) + " 训练" + str(self.t) + "次")

    def _predict(self, X):
        return X @ self.theta

    def predict(self, X):
        data = np.c_[np.ones(len(X)), X]
        return data @ self.theta
