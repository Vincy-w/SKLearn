# Scikit-Learn学习笔记

章节0：[环境配置](#0)

章节1：[决策树](#1)

章节2：[随机森林](#2)



<p id="2"></p>



### &sect;2.随机森林

**一、概述**

​	多个模型集成成为的模型叫做集成评估器（ensemble estimator），组成集成评估器的每个模型都叫做基评估器（base estimator）。通常来说，有三类集成算法：装袋法（Bagging），提升法（Boosting）和stacking。

​	装袋法核心思想是构建多个相互独立的评估器，然后对其预测进行平均或多数表决原则来决定集成评估器的结果。装袋法的代表模型就是随机森林。

**二、重要参数**

​	多数和决策树重要参数相同

1、n_estimators

​	森林中树木的数量，这个参数对随机森林模型的精确性影响是单调的，n_estimators越大，模型的效果往往越好。但是相应的，任何模型都有决策边界，n_estimators达到一定的程度之后，随机森林的精确性往往不在上升或开始波动，并且，n_estimators越大，需要的计算量和内存也越大，训练的时间也会越来越长。对于这个参数，我们是渴望在训练难度和模型效果之间取得平衡。

**三、重要属性和接口**

1、属性：

​	①.estimators_：用来查看随机森林中所有数的列表

​	②oob_score_：袋外得分，随机森林为了确保林中的每棵树都不尽相同，所以采用了对训练集进行有放回抽样的方式来不断组成信的训练集，在这个过程中，会有一些数据从来没有被随机挑选到，他们就被叫做“袋外数据”。

​	③.feature_importances_

2、接口

​	四个常用：apply, fifit, predict和score

​	特有：predict_proba接口，这个接口返回每个测试样本对应的被分到每一类标签的概率，标签有几个分类就返回几个概率。如果是二分类问题，则predict_proba返回的数值大于0.5的，被分为1，小于0.5的，被分为0。

**四、调参思想**

1、基本思想：

​	①模型太复杂或者太简单，都会让泛化误差高，我们追求的是位于中间的平衡点

​	②模型太复杂就会过拟合，模型太简单就会欠拟合

​	③对树模型和树的集成模型来说，树的深度越深，枝叶越多，模型越复杂

​	④树模型和树的集成模型的目标，都是减少模型复杂度，把模型往图像的左边移动



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
