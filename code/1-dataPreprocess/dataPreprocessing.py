# 数据预处理-机器学习的开始
# 主要使用两个包 numpy 和 pandas
# numpy 处理数学计算
# pandas 管理数据集

import numpy as np
import pandas as pd

# 步骤
# 1. 导入数据集
# 数据集通常是csv格式，使用pandas中read_csv函数可以读取本地的csv文件
dataset = pd.read_csv('data.csv')
# X是数据集中所有数据的值
X = dataset.iloc[:, :-1].values  # 第一个值表示行，第二个值标识列，这里第一个值表示所有的行，第二个值表示除了最后一列的每一列
# Y是数据集的所有标签
Y = dataset.iloc[:, 3].values  # 表示每一行的第3列

print(X)
# [['France' 44.0 72000.0]
#  ['Spain' 27.0 48000.0]
#  ['Germany' 30.0 54000.0]
#  ['Spain' 38.0 61000.0]
#  ['Germany' 40.0 nan]
#  ['France' 35.0 58000.0]
#  ['Spain' nan 52000.0]
#  ['France' 48.0 79000.0]
#  ['Germany' 50.0 83000.0]
#  ['France' 37.0 67000.0]]
print(Y)
# ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']

# 2. 处理丢失值
# SimpleImputer
from sklearn.impute import SimpleImputer

si = SimpleImputer(missing_values=np.nan, strategy="mean")  # 取均值
# 可以选用"mean", "median", "most_frequent", "constant" => 均值，中位数，众数，用fill_value替换缺失值。可用于非数值数据。
# e.g. SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=None)

si.fit(X[:, 1:])  # si.fit(X[:, 1:])
X[:, 1:] = si.transform(X[:, 1:])  # X[:, 1:3] = si.transform(X[:, 1:3]) 左闭右开

print(X)
# [['France' 44.0 72000.0]
#  ['Spain' 27.0 48000.0]
#  ['Germany' 30.0 54000.0]
#  ['Spain' 38.0 61000.0]
#  ['Germany' 40.0 63777.77777777778]
#  ['France' 35.0 58000.0]
#  ['Spain' 38.77777777777778 52000.0]
#  ['France' 48.0 79000.0]
#  ['Germany' 50.0 83000.0]
#  ['France' 37.0 67000.0]]

# 3. 类别处理
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_X = LabelEncoder()
# 将非数字标签如 France, Germany, Spain 替换为 0,1,2
# 也可以将没有数学意义的数字标签进行转换
X[:, 0] = label_X.fit_transform(X[:, 0])
print(X)

# [[0 44.0 72000.0]
#  [2 27.0 48000.0]
#  [1 30.0 54000.0]
#  [2 38.0 61000.0]
#  [1 40.0 63777.77777777778]
#  [0 35.0 58000.0]
#  [2 38.77777777777778 52000.0]
#  [0 48.0 79000.0]
#  [1 50.0 83000.0]
#  [0 37.0 67000.0]]

# 创建虚拟变量
# 将所有的变量转为binary的，即每个变量仅有0，1两个值
# 其中有三个值的变量如 France, Germany, Spain 则转换为3个虚拟变量，如果是France=>[1,0,0], Germany=>[0,1,0], Spain=>[0,0,1]
# 同理原来的第二个变量共有10个值则会产生10个虚拟变量，第三个变量也有10个值，也会产生10个虚拟变量，总共有23个虚拟变量
oe = OneHotEncoder()
X = oe.fit_transform(X).toarray()
label_Y = LabelEncoder()
Y = label_Y.fit_transform(Y)

print(len(X[0]))  # 23
print(Y)  # [0 1 0 0 1 1 0 1 0 1]

# 4. 将数据集分为训练集和测试集
from sklearn.model_selection import train_test_split

# 按照 2:8 的比例分割测试集和训练集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(len(X_train))  # 8
print(len(Y_train))  # 8
print(len(X_test))  # 2
print(len(Y_test))  # 2

# 5.特征缩放
from sklearn.preprocessing import StandardScaler

# 因为在原始的资料中，各变数的范围大不相同。对于某些机器学习的算法，若没有做过标准化，目标函数会无法适当的运作。
# 在机器学习中，我们可能要处理不同种类的资料
# 例如，音讯和图片上的像素值，这些资料可能是高维度的
# 资料标准化后会使每个特征中的数值平均变为0(将每个特征的值都减掉原始资料中该特征的平均)、标准差变为1
# 这个方法被广泛的使用在许多机器学习算法中。
sc_X = StandardScaler()
# 该方法计算数据集的均值和方差，默认都开启
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print(X_train)
