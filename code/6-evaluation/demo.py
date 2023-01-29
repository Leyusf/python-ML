import pandas as pd
from refression_evaluator import RegressionEvaluator
from classifier_evaluator import BinaryClassifierEvaluator
import numpy as np
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
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test))
c = BinaryClassifierEvaluator()
c.evaluate(y_test, y_pred)
print(c.confusion_matrix())
from sklearn.metrics import confusion_matrix
C = confusion_matrix(y_test, y_pred)
print(C)

r = RegressionEvaluator("rmse")
Y_t = np.array([1, 2, 3, 4, 5])
Y_p = np.array([2, 2, 3, 1, 4])
print(r.evaluate(Y_t, Y_p))


# k-fold 交叉验证
from sklearn.model_selection import cross_validate
classifier = LogisticRegression()
# 5折交叉验证，并返回模型
dataset = load_breast_cancer()
data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
data['cancer'] = [dataset.target_names[t] for t in dataset.target]
X = dataset.data
Y = dataset.target
X = sc.fit_transform(X)
scores = cross_validate(classifier, X, Y, cv=5, return_estimator=True)
print(scores)


