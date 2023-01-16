# 二分类逻辑回归

### 模型介绍

![](https://pic1.zhimg.com/v2-041e6fb0eb46dbca7ed1107abf108cd4_r.jpg)

输入是特征向量 $x$ ，进行加权求和，再使用 sigmoid 激活函数得到一个0到1之间的值。

逻辑回归训练的过程是通过创造基于极大似然估计原理的损失函数(Loss Function)，并利用梯度下降法更新模型中的权重，使得损失函数达到最小。

$y_i = P(y_i=1) = {1 \over 1 + e^{\theta^Tx_i}}$

### 损失函数

对于逻辑回归模型，我们希望找到一个权重向量 $\theta$，使得在训练集中模型的输出与给定的标签越接近越好，及若标签为1，则模型输出值越接近1，若标签为0，则模型输出值越接近于0。

损失函数是负对数似然值(Negative Log Likelihood, NLL)。

设定模型为 $f_{\theta}(x)=\sigma(\theta^Tx)$

损失函数为 $L(f)=\sum_nC(f(x^n),y^n)$

$C(f(x^n),y^n)=-[y^nlnf_{\theta}(x^n)+(1-y^n)ln(1-f_{\theta}(x^n))]$

推导:

假设数据是基于 $f_{\theta}(x)=f_{\theta}(C_1|x)$

前两百个数据是第一类，后三百个数据是第二类

给定一个 $\theta$, 产生这个数据的可能性是

$L(\theta)=f_{\theta}(x^1)f_{\theta}(x^2)...f_{\theta}(x^{200})(1-f_{\theta}(x^{201}))...(1-f_{\theta}(x^{500}))$

可能性越大，偏差大越小，所以要找到使得这个式子最大的 $\theta^*$

$\theta^* = argmax_{\theta} L(\theta)$

$\theta^*=argmin_{\theta} (-lnL(\theta))$

所以

$-lnL(\theta)=-lnf_{\theta}(x^1)-lnf_{\theta}(x^2)-...-lnf_{\theta}(x^{200})-ln(1-f_{\theta}(x^{201}))-...-ln(1-f_{\theta}(x^{500}))$

$=-[y^{1}lnf(x^{1})+(1-y^{1})ln(1-f(x^{1}))]-[y^{2}lnf(x^{2})+(1-y^{2})ln(1-f(x^{2}))]...-[y^{500}lnf(x^{500})+(1-y^{500})ln(1-f(x^{500}))]$

![image.png](./assets/image.png)

$=\sum_n-[y^nlnf(x^n)+(1-y^n)ln(1-f(x^n))]$

### 梯度下降法计算权重

$-lnL(\theta)=\sum_n-[y^nlnf(x^n)+(1-y^n)ln(1-f(x^n))]$

$-{\partial lnL(\theta) \over \partial \theta_i}=\sum_n-[y^n {\partial lnf(x^n) \over \partial \theta_i}+(1-y^n){\partial ln(1-f(x^n)) \over \partial \theta_i}]$

其中,根据链式法则

${\partial lnf(x^n) \over \partial \theta_i} = {\partial lnf(x^n) \over \partial z}{\partial z \over \partial \theta_i}$

${\partial ln(1-f(x^n)) \over \partial \theta_i} = {\partial ln(1-f(x^n)) \over \partial z}{\partial z \over \partial \theta_i}$


由于 $f(x^n)=\sigma(z)={1 \over 1+e^{-z}}$, $z=\theta^Tx$

${\partial lnf(x^n) \over \partial z}={\partial ln \sigma(z) \over \partial \sigma(z)}{\partial \sigma(z) \over \partial z}={1 \over \sigma(z)}\sigma(z)(1-\sigma(z))=1-\sigma(z)=（1-f(x^n)）$

${\partial z \over \partial \theta_i}=x_i$

所以

${\partial lnf(x^n) \over \partial \theta_i}=(1-f(x^n))x^n_i$

同理

${\partial ln(1-f(x^n)) \over \partial z} ={\partial ln(1-\sigma(z))\over \partial \sigma(z)}{\partial \sigma(z) \over \partial z}={-1 \over 1-\sigma(z)}{\sigma(z)(1-\sigma(z))}=-\sigma(z)$

所以

${\partial ln(1-f(x^n)) \over \partial \theta_i} =-f(x^n)x^n_i$

综上所述可知，$\theta_i$ 当前的梯度是:

$-{\partial lnL(\theta) \over \partial \theta_i}=\sum_n-(y^n-f(x^n))x^n_i$



所以更新 $\theta_i$ 使用:

这种形式叫做交叉熵(cross entropy)

$\theta_i=\theta_i-\eta\sum_n-(y^n-f(x^n))x^n_i$


### 逻辑回归为什么可以用来分类？

逻辑回归是回归的一种，由于其预测的值是一个二项分布的，也就是说预测0或1。

所以可以用来解决二分类问题。


# 总结

sigmoid 函数是: $\sigma(z)={1 \over 1+e^{-z}}$

导数形式是: $\sigma^{'}(z)=\sigma(z)(1-\sigma(z))$

损失函数是: $L(\theta)=-\sum_n[y^nlnf_{\theta}(x^n)+(1-y^n)ln(1-f_{\theta}(x^n))]$

损失函数求导（梯度）: $-{\partial lnL(\theta) \over \partial \theta_i}=\sum_n-(y^n-f(x^n))x^n_i$
