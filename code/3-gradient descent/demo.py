# 使用numpy实现梯度下降的线性回归
import matplotlib.pyplot as plt
import numpy as np

from LR_GD import LR_GD
import pandas as pd

# 1. 数据预处理
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, 1:2].values

Y = dataset.iloc[:, 2].values
# 切分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# sk LR
from sklearn.linear_model import LinearRegression

SLR = LinearRegression()
SLR.fit(X_train, Y_train)
Y_SLR = SLR.predict(X_test)
print(Y_SLR)

plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, SLR.predict(X_train), color="blue")
plt.title("SLR_train")
plt.show()

plt.scatter(X_test, Y_test, color="red")
plt.plot(X_test, Y_SLR, color="blue")
plt.title("SLR_test")
plt.show()

# GD
linearRegressor_GD = LR_GD(epochs=5000, rate=0.01)
linearRegressor_GD.fit(X_train, Y_train)
Y_GD = linearRegressor_GD.predict(X_test)
print(Y_GD)

plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, linearRegressor_GD.predict(X_train), color="blue")
plt.title("GD_train")
plt.show()

plt.scatter(X_test, Y_test, color="red")
plt.plot(X_test, Y_GD, color="blue")
plt.title("GD_test")
plt.show()
