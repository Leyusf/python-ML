# 使用numpy实现梯度下降的线性回归
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LR_GD import LR_GD
from impLR import MyLinearRegressor

# 1. 数据预处理
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, 1:2].values

Y = dataset.iloc[:, 2].values
# 切分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# sk LR
from sklearn.linear_model import LinearRegression

X_train_2 = np.c_[np.ones(len(X_train)), X_train, X_train * X_train]
X_test_2 = np.c_[np.ones(len(X_test)), X_test, X_test * X_test]

print("Test: " + str(Y_test))

SLR = LinearRegression()
SLR.fit(X_train, Y_train)
Y_Pred = SLR.predict(X_test)
print("LinearRegression:")
print(Y_Pred)
print(np.sum(np.square(Y_Pred - Y_test)) / len(X_test))

# sk LR 二次
SLR = LinearRegression()
SLR.fit(X_train_2, Y_train)
Y_Pred = SLR.predict(X_test_2)
print("LinearRegression 2次:")
print(Y_Pred)
print(np.sum(np.square(Y_Pred - Y_test)) / len(X_test))

# 等式LR
myLR = MyLinearRegressor()
myLR.fit(X_train, Y_train)
Y_Pred = myLR.predict(X_test)
print("My LinearRegression:")
print(Y_Pred)
print(np.sum(np.square(Y_Pred - Y_test)) / len(X_test))

myLR = MyLinearRegressor()
myLR.fit(X_train_2, Y_train)
Y_Pred = myLR.predict(X_test_2)
print("My LinearRegression 2次:")
print(Y_Pred)
print(np.sum(np.square(Y_Pred - Y_test)) / len(X_test))

# GD
print("GD:")
linearRegressor_GD = LR_GD(epochs=2000, rate=0.01)
linearRegressor_GD.fit(X_train, Y_train)
Y_Pred = linearRegressor_GD.predict(X_test)
print(Y_Pred)
print(np.sum(np.square(Y_Pred - Y_test)) / len(X_test))

# SGD
print("SGD:")
lR_SGD = LR_GD(epochs=2000, rate=0.01, batch_size=5)
lR_SGD.fit(X_train, Y_train)
Y_Pred = lR_SGD.predict(X_test)
print(Y_Pred)
print(np.sum(np.square(Y_Pred - Y_test)) / len(X_test))

# GD二次
print("GD 2次:")
lR_GD2 = LR_GD(epochs=200000, rate=0.0004)
lR_GD2.fit(X_train_2, Y_train)
Y_Pred = lR_GD2.predict(X_test_2)
print(Y_Pred)
print(np.sum(np.square(Y_Pred - Y_test)) / len(X_test))




