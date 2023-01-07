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
        # 注意numpy类型转换错误
        invX = np.linalg.inv((XT @ data).astype(np.float64))
        # 计算权重
        self.theta = invX @ XT @ Y

    def predict(self, X):
        data = np.c_[np.ones(len(X)), X]
        return data @ self.theta
