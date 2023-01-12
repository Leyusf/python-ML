# 使用numpy实现随机梯度下降的线性回归
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from testLR import LR_GD

# 1. 数据预处理
dataset = pd.read_csv("winequality-white.csv")
X = dataset.iloc[:, :-1].values
Y = dataset["quality"].values
# 切分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

print(Y_test)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
print(Y_pred)
print(np.sum(np.square(Y_pred - Y_test)) / len(X_test))

# batch GD
# print("\nGD:")
# lR_GD = LR_GD(epochs=20000, rate=0.000001)
# lR_GD.fit(X_train, Y_train)
# Y_Pred = lR_GD.predict(X_test)
# print(Y_Pred)
# print(lR_GD.meanSquareError(X_test, Y_test))

# SGD
# print("\nSGD:")
# lR_SGD = LR_GD(epochs=20000, rate=0.000001, batch_size=10)
# lR_SGD.fit(X_train, Y_train)
# Y_Pred = lR_SGD.predict(X_test)
# print(Y_Pred)
# print(lR_SGD.meanSquareError(X_test, Y_test))

# 自适应学习率
# print("\n自适应学习率:")
# linearRegressor_GD = LR_GD(epochs=20000, rate=0.000001, alr=True)
# linearRegressor_GD.fit(X_train, Y_train)
# Y_Pred = linearRegressor_GD.predict(X_test)
# print(Y_Pred)
# print(linearRegressor_GD.meanSquareError(X_test, Y_test))

# Adagrad
print("\nAdagrad:")
linearRegressor_GD = LR_GD(epochs=20000, rate=0.01, alg="rmsprop", batch_size=100)
linearRegressor_GD.fit(X_train, Y_train)
Y_Pred = linearRegressor_GD.predict(X_test)
print(Y_Pred)
print(linearRegressor_GD.meanSquareError(X_test, Y_test))
