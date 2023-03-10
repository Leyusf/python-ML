import numpy as np


class MyLinearRegressor:

    def __init__(self):
        self.theta = None

    def fit(self, X, Y):
        # theta = inv(X^T@X)@X^T@y
        data = np.c_[np.ones(len(X)), X]
        # 添加单位向量用来接受偏置
        XT = data.T
        # 计算矩阵的逆
        invX = np.linalg.pinv(XT @ data)
        # 计算权重
        self.theta = invX @ XT @ Y

    def predict(self, X):
        data = np.c_[np.ones(len(X)), X]
        return data @ self.theta
