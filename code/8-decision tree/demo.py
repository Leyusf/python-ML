import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from DecisionTreeRegressorImp import DecisionTreeRegression

# 1. 数据预处理
# boston房价预测数据集
from sklearn.datasets import load_diabetes
housing_boston = load_diabetes()
X = housing_boston.data     # data
Y = housing_boston.target   # label
# 切分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
tree = DecisionTreeRegressor(max_depth=4, min_samples_split=5)
tree.fit(X_train, Y_train)

myTree = DecisionTreeRegression(max_depth=4, min_samples_leaf=5)
myTree.fit(X_train, Y_train)
pred = myTree.predict(X_test)


# 计算mse之和
def mean_squared_error(y_test, y_pred):
    mse = np.mean(np.square(y_test - y_pred))
    return mse


print(mean_squared_error(Y_test, np.around(myTree.predict(X_test))))
print(mean_squared_error(Y_test, np.around(tree.predict(X_test))))
