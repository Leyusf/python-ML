import numpy as np


class RegressionEvaluator:

    def __init__(self, alg=""):
        self.Y_p = None
        self.Y = None
        if alg == "" or alg == "mse":
            self.error = self.MSE
        elif alg == "rss":
            self.error = self.RSS
        elif alg == "ase":
            self.error = self.ASE
        elif alg == "cof":
            self.error = self.R
        elif alg == "mas":
            self.error = self.MAS
        elif alg == "rmse":
            self.error = self.RMSE
        else:
            print("错误的算法: " + alg)

    def RSS(self):
        # 残差平方和(Residual Sum of Squares)
        e = self.Y - self.Y_p
        err = np.sum(e @ e)
        return err

    def ASE(self):
        # 绝对误差和(Sum of Absolute Errors)
        e = np.sum(self.Y - self.Y_p)
        return e

    def R(self):
        # 决定系数(Coefficient of determination)
        Y_a = np.sum(self.Y)/len(self.Y)
        me = self.Y - Y_a
        r = 1 - (self.RSS()/np.sum(me @ me))
        return r

    def MSE(self):
        # 均方误差(Mean Squared Error)
        return self.RSS()/len(self.Y)

    def MAS(self):
        # 绝对平均误差(Mean Absolute Error)
        return self.ASE()/len(self.Y)

    def RMSE(self):
        # 均方根误差(Root Mean Squared Error)
        return pow(self.MSE(), 0.5)

    def evaluate(self, Y, Y_p):
        self.Y = Y
        self.Y_p = Y_p
        return self.error()



