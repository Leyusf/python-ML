import numpy as np

# numpy中向量是横向量
# 1. 创建一个数组，包含数字0-15
a = np.arange(16)
print("创建一个数组，包含数字0-15")
print(a)

# 2. 将其变换为 4x4 的矩阵
a = a.reshape(4, 4)
print("\n将其变换为 4x4 的矩阵")
print(a)

# 3. 矩阵点乘2
print("\n点乘2")
print(a * 2)

# 4. 矩阵转置
print("\n矩阵转置")
print(a.T)
print()
print(a.transpose())

# 5. 矩阵均值
print("\n矩阵均值")
print(a.mean())

# 6. 生成 0到pi 之间50个等间隔的数字
x = np.linspace(0, np.pi)  # 默认50个数字，可以使用num来指定
print("\n生成 0到pi 之间50个等间隔的数字")
print(len(x))
print(x)

# 7. 进行cos计算
y = np.cos(x)
print("\n进行cos计算")
print(y)

# 8. 获取子矩阵
# 获取索引3-9的值
print("\n获取索引3-9的值")
print(x[3:10])
# 获取索引0-9的值
print("\n获取索引0-9的值")
print(x[:10])
# 获取索引10以及索引10之后的值
print("\n获取索引10以及索引10之后的值")
print(x[10:])
# 获取前10个值
print("\n获取前10个值")
print(x[:10])
# 获取所有的值
print("\n获取所有的值")
print(x[:])

# 9. 获取列，使用 4x4矩阵a
print("\n矩阵a")
print(a)
# 获取第三列
print("\n获取第三列")
print(a[:, 2])
# 获取中间两列
print("\n获取中间两列")
print(a[:, 1:3])
# 获取内部子矩阵
print("\n获取内部子矩阵")
print(a[1:3, 1:3])


# 9. 布尔值掩码，使用y
# 可以获取有或者无
print("\n矩阵y")
print(y)
print("\n布尔值转换，使用y")
mask = y > 0
print(mask)
print(x[mask])
