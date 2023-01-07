import matplotlib.pyplot as plt
import pandas as pd
# 使用numpy实现的LinearRegressor
from impLR import MyLinearRegressor

# 1. 数据预处理
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, :1].values


Y = dataset.iloc[:, 1].values
# 切分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# 计算权重
myLR = MyLinearRegressor()
myLR.fit(X_train, Y_train)
print(myLR.theta)
# 预测
Y_pred1 = myLR.predict(X_test)

# 对比
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
Y_pred2 = regressor.predict(X_test)

print(Y_pred1)
print(Y_pred2)

# 4. 可视化
# 训练集
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, myLR.predict(X_train), color="blue")
plt.show()

# 测试集
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_test, Y_pred1, color="blue")
plt.show()
