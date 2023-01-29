import numpy as np


class BinaryClassifierEvaluator:
    def __init__(self):
        self.FN = None
        self.FP = None
        self.TN = None
        self.TP = None
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
        self.TP = np.sum(self.Y & self.Y_p)
        # 计算真负样本数量
        self.TN = len(self.Y) - np.sum(self.Y | self.Y_p)
        # 计算假正样本数量
        self.FP = np.sum(self.Y_p) - self.TP
        # 计算假负样本数量
        self.FN = len(self.Y_p) - self.TP - self.FP - self.TN

    def accuracy_rate(self):
        # 正确率
        return (self.TP + self.TN) / (self.P + self.N)

    def FP_error_rate(self):
        # 假正率
        return self.FP / (self.P + self.N)

    def FN_error_rate(self):
        # 假负率
        return self.FN / (self.P + self.N)

    def specificity(self):
        # 特异性 (真正率)
        return 1 - self.FP_error_rate()

    def sensitivity(self):
        # 敏感性 (真正率,召回率)
        return 1 - self.FN_error_rate()

    def precision(self):
        # 精度
        return self.TP / (self.TP + self.FP)

    def recall(self):
        # 召回率
        return self.sensitivity()

    def F1_score(self):
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())

    def confusion_matrix(self):
        return np.array([
            [self.TN, self.FP],
            [self.FN, self.TP]
        ])
