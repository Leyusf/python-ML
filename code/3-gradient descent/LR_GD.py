import numpy as np


class LR_GD:

    # epochs 是最大训练轮数
    # rate 是学习率
    # e 是允许的误差
    def __init__(self, epochs, rate=0.01, e=1e-7):
        self.theta = None
        self.epochs = epochs
        self.rate = rate
        self.e = e

    def gradient(self, X, Y):
        # y = b0 + b1x1 + b2x2 + ... + bnxn
        return -2 / len(X) * X.T @ (Y - X @ self.theta)

    def fit(self, X, Y):
        Y = Y.reshape(len(Y), 1)
        data = np.c_[np.ones(len(X)), X]
        t = 0
        mse = np.inf
        self.theta = np.c_[np.zeros(len(data[0]))]
        while t < self.epochs and mse > self.e:
            v = self.rate * self.gradient(data, Y)
            self.theta = self.theta - v
            t += 1
            mse = np.ones(len(Y)) @ ((Y - self.predict(X)) * (Y - self.predict(X)))
        print("训练完成, MSE=" + str(mse) + " 训练" + str(t) + "次")

    def predict(self, X):
        data = np.c_[np.ones(len(X)), X]
        return data @ self.theta
