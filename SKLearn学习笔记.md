# Scikit-Learn学习笔记

章节0：[环境配置](#0)

章节1：[决策树](#1)



<p id="1"></p>



### &sect;1.决策树

**一、重要参数**

1、criterion

​	用于计算不纯度（不纯度越低越好，且子节点的不纯度一定低于父节点）。sklearn提供两种计算选择（实际使用中两种效果基本相同）：

​	1）entropy，信息熵

​	2）gini，基尼系数

2、random_state & splitter

​	random_state用于设置分支中随机模式的参数。

​	splitter用于控制决策树中的随机选项，有两种输入值，输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝，输入“random"，决策树在分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。这也是防止过拟合的一种方式。

3、剪枝参数

​	1）max_depth 限制树的最大深度

​	2）min_samples_leaf & min_samples_split

​		min_samples_leaf限定，一个节点在分支后的每个子节点都必须至少min_samples_leaf个训练样	本，否则不分枝。

​		min_samples_split限定，一个节点至少包含min_samples_split个训练样本，才被允许分支

​	3）max_features & min_impurity_decrease

​		max_features限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。

​		min_impurity_decrease限制信息增益的大小，信息增益小于设定数值的分枝不会发生。

**二、一个属性**

feature_importances_：可查看每个特征对模型的重要性

**三、四个接口**

1、建模流程

​	`from sklearn import tree #导入需要的模块`

​    `clf = tree.DecisionTreeClassifier()   #实例化`

​	`clf = clf.fit(X_train,y_train) #用训练集数据训练模型`

​	`result = clf.score(X_test,y_test) #导入测试集，从接口中调用需要的信息`

2、`#apply返回每个测试样本所在的叶子节点的索引`

  	`clf.apply(Xtest)`

​	  `#predict返回每个测试样本的分类/回归结果`

​	  `clf.predict(Xtest)`



<p id="0"></p>



### &sect;0.环境配置

**一、Jupyter lab**

1、Anaconda

参考链接：[Anaconda介绍、安装及使用教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32925500)

2、Jupyter使用方法简介

参考链接：[简单了解JupyterLab的使用方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/383005827)

3、修改JupyterLab默认路径

参考链接：[修改Jupyter Lab、Jupyter Notebook的工作目录_奶茶可可的博客-CSDN博客_jupyter 修改目录](https://blog.csdn.net/xing09268/article/details/123919230?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-123919230-blog-126015415.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-123919230-blog-126015415.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=2)

**二、graphviz**

安装并配置环境，但在Jupyter中查看graphviz的版本时报错

解决方法：

在anaconda prompt界面输入

conda install graphviz

conda install python-graphviz

**三、我的库和版本**

python 3.9.13

sklearn 1.0.2

Graphriz 0.20.1

Numpy 1.21.5

Pandas 1.4.4

Mauplotlib 3.5.2

SciPy 1.9.1
