import numpy as np

from LR_GD import LR_GD
import pandas as pd
# 展示自适应学习率

dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, 1:2].values

Y = dataset.iloc[:, 2].values
# 切分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

print("Test: " + str(Y_test))

print("GD固定学习率:")
linearRegressor_GD = LR_GD(epochs=9000, rate=0.01)
linearRegressor_GD.fit(X_train, Y_train)
Y_Pred = linearRegressor_GD.predict(X_test)
print(Y_Pred)
print(linearRegressor_GD.meanSquareError(X_test, Y_test))

print("\nGD自适应学习率:")
linearRegressor_GD = LR_GD(epochs=9000, rate=0.1, alg="adaptive")
linearRegressor_GD.fit(X_train, Y_train)
Y_Pred = linearRegressor_GD.predict(X_test)
print(Y_Pred)
print(linearRegressor_GD.meanSquareError(X_test, Y_test))

X_train_2 = np.c_[np.ones(len(X_train)), X_train, X_train * X_train]
X_test_2 = np.c_[np.ones(len(X_test)), X_test, X_test * X_test]


print("\nGD固定学习率:")
linearRegressor_GD = LR_GD(epochs=100000, rate=0.0004)
linearRegressor_GD.fit(X_train_2, Y_train)
Y_Pred = linearRegressor_GD.predict(X_test_2)
print(Y_Pred)
print(linearRegressor_GD.meanSquareError(X_test_2, Y_test))

print("\nGD自适应学习率:")
linearRegressor_GD = LR_GD(epochs=100000, rate=0.001, alg="adaptive")
linearRegressor_GD.fit(X_train_2, Y_train)
Y_Pred = linearRegressor_GD.predict(X_test_2)
print(Y_Pred)
print(linearRegressor_GD.meanSquareError(X_test_2, Y_test))


