# 分类

预测目标变量的类别。

### 分类步骤：

1. 找到一个函数 $f(x)$ 使得
   $ x= \{ {f(x) > \tau, output = class A \atop otherwise, output = class B}$
2. 评估(计算损失函数)
   $L(f)=\sum_n \delta(f(x^n) \neq y^n)$
3. 最小会损失函数
   $min(L(f))$

### 概率分类器：

二次判别分析（QDA）、线性判别分析（LDA）和朴素贝叶斯

### 概率

![image.png](./assets/image.png)

1. 先验概率: 从两个盒子中选一个球的概率
   $P(B_1)={5 \over 13}$
   $P(B_2)={8 \over 13}$
2. 条件概率: 已知选择的盒子时取得白球和红球的概率
   $P(white|B_1)={2 \over 5} \ P(red|B_1)={3 \over 5}$
   $P(white|B_2) = { 6 \over 8} \ P(red|B_2)={2 \over 8}$
3. 后验概率: 已知取得一个白球，从 $B_1$ 中取球的概率
   $P(B_1|white)={P(B_1)P(white|B_1) \over P(B_1)P(white|B_1) + P(B_2)P(white|B_2)}$
4. 概率分类原理: 给定一个x，预测它的类别 $C$
   $P(C_1|x) = {P(C_1)P(x|C_1) \over P(C_1)P(x|C_1) + P(C_2)P(x|C_2) + ... + P(C_n)P(x|C_n)}$


### 似然

从概率推参数的过程。

假设 $\theta$ 是环境参数（条件）， $X$ 是事件发生的结果。

计算概率 $P(X|\theta)$ ,在条件为 $\theta$ 下发生 $X$ 的概率。

计算似然 $L(\theta|X)$ ,在 $X$ 发生时，推断 $\theta$ 的值。


### 最大似然估计(Maximum Likelihood Estimate)

根据已知样本的结果，反推最大概率 $\theta$。

e.g.

在装有无限个红球和白球的盒子中随机取5个球，计算一次取球时，取得红球和白球的概率。

设 $P(red)=\theta \ \ \ \ \ \ \ \ \ P(white)=1-\theta$

抽取的样本为:

R W R R W

1. 获得这个样本的概率为: $L(\theta)=\theta^3(1-\theta)^2$
2. 求使得 $L(\theta)$ 最大的 $\theta$
   $ln L(\theta) =3ln\theta+2ln(1-\theta)$
   ${\partial lnL(\theta) \over \partial \theta} = {3\over\theta}-{2\over1-\theta} = 0$
   $\theta={3\over5}$


### 高斯分布

$f_{\mu,\Sigma}(x)={1 \over (2\pi)^{D/2}}{1\over|\Sigma|^{1/2}}e^{-{1\over 2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$

其中,

$x$ 是长度为 $D$ 的向量

$\mu$ 是均值期望

$\Sigma$ 是协方差矩阵(方差组成的矩阵)

$f_{\mu,\Sigma}(x)$ 概率密度



### 估计 $\mu$ 和 $\Sigma$

最大似然

$L(\mu,\Sigma)=f_{\mu,\Sigma}(x^1)f_{\mu,\Sigma}(x^2)f_{\mu,\Sigma}(x^3)...f_{\mu,\Sigma}(x^n)$

使其最大的 $\mu$ 和 $\Sigma$:

$\mu,\Sigma=max(L(\mu,\Sigma))$

$\mu={\sum^n_{i=1}x^i\over n}$

$\Sigma={\sum^n_{i=1}(x^i-\mu)(x^i-\mu)^T \over n}$


### QDA

$P(x| C_1)=f_{\mu_1,\Sigma_1}(x)$

$P(x|C_2)=f_{\mu_2,\Sigma_2}(x)$


### LDA

不同的类可能共享协方差矩阵 $Σ$

$L(\mu_1,\mu_2\Sigma)=f_{\mu_1,\Sigma}(x^1)f_{\mu_1,\Sigma}(x^2)...f_{\mu_1,\Sigma}(x^n) \times f_{\mu_2,\Sigma}(x^{n+1})f_{\mu_2,\Sigma}(x^{n+2})...f_{\mu_2,\Sigma}(x^{n+m})$

$\mu_1 = {\sum^n_{i=1}x_i \over n}$

$\mu_2={\sum^{m+n}_{i=n+1}x^i \over m}$

$\Sigma={n\over n+m}\Sigma_1 + {m \over n+m}\Sigma_2$


### Naïve Bayes(朴素贝叶斯)

当x的所有特征都是独立的，使用朴素贝叶斯。

![image.png](./assets/1673637220767-image.png)

$P(x|C_1) = \prod^n_{i=1} P(x_i|C_1)$
