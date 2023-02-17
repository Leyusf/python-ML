import numpy as np


# 信息增益
def gain(y, y_left, y_right):
    # gain选最大的，所以要加负号
    return - (shannonEntropy(y) - (len(y_left) / len(y) * shannonEntropy(y_left) + len(y_right) / len(y) * shannonEntropy(y_right)))

# 香农信息熵
def shannonEntropy(y):
    _, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    e = -np.sum(prob * np.log2(prob))
    return e


# 基尼纯度
def gini(y):
    classes = np.unique(y)
    prob = []
    n = len(y)
    for c in classes:
        prob.append(len(y[c == y]) / n)
    return 1 - np.sum(np.square(prob))


# 基尼指数
def gini_index(y, y_left, y_right):
    return (len(y_left) / len(y)) * gini(y_left) + (len(y_right) / len(y)) * gini(y_right)


class Node:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx  # 该节点的划分特征
        self.threshold = threshold  # 该节点的划分阈值
        self.value = value  # 该节点的预测值（仅用于叶子节点）
        self.left = left  # 该节点的左子节点
        self.right = right  # 该节点的右子节点


# 分割数据集
def split_dataset(X, y, feature_index, threshold):
    left_X = X[X[:, feature_index] < threshold]
    left_y = y[X[:, feature_index] < threshold]
    right_X = X[X[:, feature_index] >= threshold]
    right_y = y[X[:, feature_index] >= threshold]
    return left_X, left_y, right_X, right_y


# 计算mse之和
def sum_squared_error(y, left_y, right_y):
    left_sse = np.sum(np.square(left_y - np.mean(left_y)))
    right_sse = np.sum(np.square(right_y - np.mean(right_y)))
    return left_sse + right_sse


def classifier(y):
    counts = np.bincount(y)
    return Node(value=np.argmax(counts))


def regressor(y):
    return Node(value=np.mean(y))


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, std="gini", target="classify"):
        self.max_depth = np.inf
        if max_depth:
            self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        if std == "gini":
            self.measure = gini_index
        elif std == "gain":
            self.measure = gain
        elif std == "gain_ratio":
            self.measure = self.gain_ratio
        elif std == "sse":
            self.measure = sum_squared_error
        else:
            print("错误的指标")
        if target == "classify":
            self.assignValue = classifier
        elif target == "regress":
            self.measure = sum_squared_error
            self.assignValue = regressor

    def measure(self):
        pass

    def assignValue(self):
        pass


    # 增益率
    def gain_ratio(self, y, y_left, y_right):
        # 计算固有值
        _, counts = np.unique(self.slice, return_counts=True)
        p = counts / np.sum(counts)
        iv = -np.sum(p * np.log2(p))
        # 增益率选最大的，所以要加负号，但是gain已经加了负号
        return gain(y, y_left, y_right) / iv

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def build_tree(self, X, y, depth):
        # 如果深度达到最大或者没有足够的样本
        if len(set(y)) == 1 or self.max_depth < depth or len(X) < self.min_samples_split:
            return self.assignValue(y)

        best_feature_idx, best_threshold = self.get_best_split(X, y)

        X_left, y_left, X_right, y_right = split_dataset(X, y, best_feature_idx, best_threshold)
        left = self.build_tree(X_left, y_left, depth + 1)
        right = self.build_tree(X_right, y_right, depth + 1)

        return Node(best_feature_idx, best_threshold, left=left, right=right)

    def get_best_split(self, X, y):
        best_feature_idx, best_threshold, best_std = None, None, float('inf')
        m, n = X.shape
        # 对每个样本的每个特征分隔，选择std最小的分割点
        for feature_idx in range(n):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                y_left = y[X[:, feature_idx] < threshold]
                y_right = y[X[:, feature_idx] >= threshold]
                self.slice = X[:, feature_idx]
                std = self.measure(y, y_left, y_right)
                if std < best_std:
                    best_feature_idx, best_threshold = feature_idx, threshold
                    best_std = std

        return best_feature_idx, best_threshold

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def predict_one(self, x):
        node = self.tree
        while node.left:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
