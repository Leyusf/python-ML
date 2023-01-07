import numpy as np
import pandas as pd

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 矩阵点成
print(a @ b)
print(a.dot(b))

# 举证叉乘
print(a * b)
print(np.multiply(a, b))

# 使用pands读取csv文件
df = pd.read_csv("winequality-white.csv")
# 获取第3行
print(df[2:3])
print()
# 获取第3-4行
print(df.loc[2:3])
# 获取第3行
print(df.iloc[2:3])

# 线性回归
# 预测质量

X = df.iloc[:, :-1].values
Y = df["quality"].values
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# 使用skl
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
print(Y_pred)
print(Y_test)

e = Y_pred - Y_test
print("平均误差是： " + str(e @ e.T / len(X_test)))

# 使用自己实现的
from impLR import MyLinearRegressor

myLR = MyLinearRegressor()
myLR.fit(X_train, Y_train)
Y_pred = myLR.predict(X_test)
print(Y_pred)
print(Y_test)

e = Y_pred - Y_test
print("平均误差是： " + str(e @ e.T / len(X_test)))
