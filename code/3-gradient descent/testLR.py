import numpy as np
import random


class LR_GD:

    # epochs 是最大训练轮数
    # rate 是学习率
    # e 是允许的误差
    def __init__(self, epochs, rate=0.01, e=1e-6, batch_size=0, alg="", p=0.9):
        self.theta = None
        self.epochs = epochs
        self.rate = rate
        self.e = e
        self.t = 0
        self.list = []
        self.batch_size = batch_size
        self.sigma = None
        self.mse = np.inf
        self.ep = 1e-10
        self.p = p
        if alg == "":
            self.Rate = self.simpleRate
        elif alg == "adaptive":
            self.Rate = self.adaptiveRate
        elif alg == "adagrad":
            self.Rate = self.adagradRate
        elif alg == "rmsprop":
            self.Rate = self.RMSpropRate
        else:
            print("错误的算法: " + alg)
            self.Rate = self.simpleRate

    def simpleRate(self, g):
        return self.rate

    def adaptiveRate(self, g):
        return self.rate / pow(self.t + 1, 0.5)

    def adagradRate(self, g):
        self.sigma += g
        return self.rate / (pow(self.sigma * self.sigma, 0.5) + self.ep)

    def RMSpropRate(self, g):
        self.sigma += pow(self.sigma * self.p + (1 - self.p) * (g**2), 0.5)
        return self.rate / (pow(self.sigma * self.sigma, 0.5) + self.ep)

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
        self.rate = np.c_[np.ones(len(data[0]))] * self.rate
        self.mse = np.inf
        self.theta = np.c_[np.zeros(len(data[0]))]
        self.sigma = np.c_[np.zeros(len(data[0]))]
        while self.t < self.epochs and self.mse > self.e:
            # 打乱索引
            batch_X, batch_Y = self.getData(data, Y)
            g = self.gradient(batch_X, batch_Y)
            v = self.Rate(g) * g
            self.theta = self.theta - v
            self.t += 1
        self.mse = self.meanSquareError(X, Y)
        print("训练完成, MSE=" + str(self.mse) + " 训练" + str(self.t) + "次")

    def meanSquareError(self, X, Y):
        Y = Y.reshape(len(Y), 1)
        error = Y - self.predict(X)
        return np.ones(len(X)) @ (error * error) / len(X)

    def predict(self, X):
        data = np.c_[np.ones(len(X)), X]
        return data @ self.theta
