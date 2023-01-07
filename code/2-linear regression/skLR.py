# 使用sklearn
import matplotlib.pyplot as plt
import pandas as pd

# 任务根据学生的学习时间推断分数
# Hours, Scores

# 1. 数据预处理
dataset = pd.read_csv("data.csv")
# 切分标签和特征
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values
print(X)
print(Y)
# 切分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# 2. 使用线性回归模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 3. 预测
Y_pred = regressor.predict(X_test)
print(Y_pred)

# 4. 可视化
# 训练集
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.show()

# 测试集
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, Y_pred, color='blue')
plt.show()
