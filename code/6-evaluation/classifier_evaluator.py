import numpy as np


class BinaryClassifierEvaluator:
    def __init__(self):
        self.N = None
        self.P = None
        self.Y = None
        self.Y_p = None

    def evaluate(self, Y, Y_p):
        self.Y = Y
        self.Y_p = Y_p
        # 正样本数量和负样本数量
        self.P = np.sum(self.Y)
        self.N = len(Y) - self.P
        # 计算真正样本数量
        TP = np.sum(self.Y & self.Y_p)
        # 计算真负样本数量
        TN = len(self.Y) - np.sum(self.Y | self.Y_p)



r = BinaryClassifierEvaluator()
Y = np.array([1, 1, 0, 0])
Y_p = np.array([1, 0, 1, 0])
r.evaluate(Y, Y_p)
