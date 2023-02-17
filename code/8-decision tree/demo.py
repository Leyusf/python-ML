import numpy as np
# 1. 数据预处理
# boston房价预测数据集
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from DecisionTree import DecisionTree

housing_boston = load_diabetes()
X = housing_boston.data     # data
Y = housing_boston.target   # label
# 切分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
tree = DecisionTreeRegressor(min_samples_split=5)
tree.fit(X_train, Y_train)

myTree = DecisionTree(min_samples_split=5, target="regress")
myTree.fit(X_train, Y_train)
pred = myTree.predict(X_test)


# 计算mse之和
def mean_squared_error(y_test, y_pred):
    mse = np.mean(np.square(y_test - y_pred))
    return mse


print(mean_squared_error(Y_test, np.around(myTree.predict(X_test))))
print(mean_squared_error(Y_test, np.around(tree.predict(X_test))))
