import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
data['cancer'] = [dataset.target_names[t] for t in dataset.target]
X = dataset.data
Y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from LogisticRefressionImpl import MyLogisticRegressor
LR = MyLogisticRegressor(1500, 0.1)
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
print(y_pred)
count_right = sum(y_pred == y_test) * 1.0 / len(y_pred)
print(count_right)

y_train = (y_train == 1) * 0.8 + 0.1
y_test = (y_test == 1) * 0.8 + 0.1

Z_train = np.log(y_train / (1 - y_train))
Z_test = np.log(y_test / (1 - y_test))
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Z_train)
y_pred = (lr.predict(X_test) > 0) * 0.8 + 0.1
print(y_pred)
count_right = sum(y_pred == y_test) * 1.0 / len(y_test)
print(count_right)

from LR_GD import LR_GD
mlr = LR_GD(1500, 0.01)
mlr.fit(X_train, Z_train)
y_pred = (lr.predict(X_test) > 0) * 0.8 + 0.1
print(y_pred)
count_right = sum(y_pred == y_test) * 1.0 / len(y_test)
print(count_right)
