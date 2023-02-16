import numpy as np


# 香农信息熵
def shannonEntropy(prob):
    e = -1 * np.sum(prob * np.log(prob))
    return e


# 分割数据集
def split_dataset(X, y, feature_index, threshold):
    left_X = X[X[:, feature_index] < threshold]
    left_y = y[X[:, feature_index] < threshold]
    right_X = X[X[:, feature_index] >= threshold]
    right_y = y[X[:, feature_index] >= threshold]
    return left_X, left_y, right_X, right_y


# 计算mse之和
def sum_squared_error(left_y, right_y):
    left_sse = np.sum(np.square(left_y - np.mean(left_y)))
    right_sse = np.sum(np.square(right_y - np.mean(right_y)))
    return left_sse + right_sse


def get_best_split(X, y):
    best_feature, best_threshold, best_mse = None, None, float('inf')
    # 对每个样本的每个特征分隔，选择mse最小的分割点
    for feature_index in range(X.shape[1]):
        for value in np.unique(X[:, feature_index]):
            # 切分
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, value)
            if len(left_X) == 0 or len(right_X) == 0:
                continue
            # 计算mse
            mse = sum_squared_error(left_y, right_y)
            if mse < best_mse:
                best_feature, best_threshold, best_mse = feature_index, value, mse
    return best_feature, best_threshold


class DecisionTreeRegression:
    def __init__(self, max_depth=np.inf, min_samples_leaf=2):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=1)

    def predict(self, X):
        return np.array([self.predict_tree(x, self.tree) for x in X])

    def build_tree(self, X, y, depth):
        # 如果深度达到最大或者没有足够的样本
        if depth > self.max_depth or len(X) < self.min_samples_leaf:
            return np.mean(y)
        # 找到最优的分割点使得两边的mse最小
        best_feature, best_threshold = get_best_split(X, y)

        # 没有数据或没有特征时
        if best_feature is None or best_threshold is None:
            return np.mean(y)
        # 切分
        left_X, left_y, right_X, right_y = split_dataset(X, y, best_feature, best_threshold)

        tree = {
            'feature_index': best_feature,
            'threshold': best_threshold,
            'left': self.build_tree(left_X, left_y, depth + 1),
            'right': self.build_tree(right_X, right_y, depth + 1)
        }
        return tree

    def predict_tree(self, x, tree):
        if isinstance(tree, float):
            return tree
        feature_index, threshold = tree['feature_index'], tree['threshold']
        if x[feature_index] < threshold:
            return self.predict_tree(x, tree['left'])
        else:
            return self.predict_tree(x, tree['right'])
