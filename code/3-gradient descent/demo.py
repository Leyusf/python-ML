# 使用numpy实现梯度下降的线性回归
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from LR_GD import LR_GD
from impLR import MyLinearRegressor

# 1. 数据预处理
dataset = pd.read_csv("winequality-white.csv")
X = dataset.iloc[:, :-1].values

Y = dataset.iloc[:, -1].values
# 切分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# sk LR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("Test: " + str(Y_test))

SLR = LinearRegression()
SLR.fit(X_train, Y_train)
Y_Pred = SLR.predict(X_test)
print("LinearRegression:")
print(Y_Pred)
print(mean_squared_error(Y, Y_Pred))

# 等式LR
myLR = MyLinearRegressor()
myLR.fit(X_train, Y_train)
Y_Pred = myLR.predict(X_test)
print("My LinearRegression:")
print(Y_Pred)
print(np.sum(np.square(Y_Pred - Y_test)) / len(X_test))

# GD
print("GD:")
linearRegressor_GD = LR_GD(epochs=2000, rate=0.001)
linearRegressor_GD.fit(X_train, Y_train)
Y_Pred = linearRegressor_GD.predict(X_test)
print(Y_Pred)
print(linearRegressor_GD.meanSquareError(X_test, Y_test))

# SGD
print("SGD:")
lR_SGD = LR_GD(epochs=2000, rate=0.001, batch_size=5)
lR_SGD.fit(X_train, Y_train)
Y_Pred = lR_SGD.predict(X_test)
print(Y_Pred)
print(lR_SGD.meanSquareError(X_test, Y_test))




