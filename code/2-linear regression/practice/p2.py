# 有非数值特征的数据集
# 预测profit
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from impLR import MyLinearRegressor

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
# 将类别数字化
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# 舍弃一个虚拟变量，躲避虚拟变量陷阱
X = X[:, 1:]
print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(regressor.predict(X_test))

# 使用自己的
myLR = MyLinearRegressor()
myLR.fit(X_train, Y_train)
print(myLR.predict(X_test))
