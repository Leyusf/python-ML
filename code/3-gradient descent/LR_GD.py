import numpy as np


class LR_GD:

    # epochs 是最大训练轮数
    # rate 是学习率
    # e 是允许的误差
    def __init__(self, epochs, rate=0.01, e=1e-7, alr=False):
        self.theta = None
        self.epochs = epochs
        self.rate = rate
        self.e = e
        self.alr = alr
        self.t = 0

    def Rate(self):
        # 自适应学习率
        if self.alr:
            return self.rate / pow(self.t + 1, 0.5)
        return self.rate

    def gradient(self, X, Y):
        # y = b0 + b1x1 + b2x2 + ... + bnxn
        return -2 / len(X) * X.T @ (Y - X @ self.theta)

    def fit(self, X, Y):
        self.t = 0
        Y = Y.reshape(len(Y), 1)
        data = np.c_[np.ones(len(X)), X]
        mse = np.inf
        self.theta = np.c_[np.zeros(len(data[0]))]
        while self.t < self.epochs and mse > self.e:
            v = self.Rate() * self.gradient(data, Y)
            self.theta = self.theta - v
            self.t += 1
            mse = np.ones(len(Y)) @ ((Y - self.predict(X)) * (Y - self.predict(X)))
        print("训练完成, MSE=" + str(mse) + " 训练" + str(self.t) + "次")

    def predict(self, X):
        data = np.c_[np.ones(len(X)), X]
        return data @ self.theta
