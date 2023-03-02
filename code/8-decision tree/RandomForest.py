import numpy as np

from DecisionTree import DecisionTree

"""
选择数据集以及特征的子集
"""
def select_X(X, y, X_size, features_size):
    m, n = X.shape
    drop_X = []
    drop_features = []
    for sample in range(m):
        a = np.random.uniform()
        if a > X_size:
            drop_X.append(sample)
    sub_X = np.delete(X, drop_X, axis=0)
    sub_y = np.delete(y, drop_X, axis=0)
    for feature in range(n):
        a = np.random.uniform()
        if a > features_size:
            drop_features.append(feature)
    sub_X = np.delete(sub_X, drop_features, axis=1)
    return sub_X, sub_y


class RandomForest:
    def __init__(self, max_depth=None, min_samples_split=2, std="gini", target="classify", max_trees=10):
        self.forest = []
        self.target = target
        for i in range(max_trees):
            self.forest.append(
                DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, std=std, target=target))

    def fit(self, X, y, X_size=0.5, features_size=0.8):
        for tree in self.forest:
            sub_X, sub_y = select_X(X, y, X_size, features_size)
            tree.fit(sub_X, sub_y)

    def predict(self, X):
        results = [tree.predict(X) for tree in self.forest]
        if self.target == "regress":
            return np.mean(results, axis=0)
        elif self.target == "classify":
            res = []
            for result in results:
                counts = np.bincount(result)
                res.append(np.argmax(counts))
            return res
