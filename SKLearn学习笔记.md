# Scikit-Learn学习笔记

章节0：[环境配置](#0)

章节1：[决策树](#1)

章节2：[随机森林](#2)

章节3：[预处理和特征工程](#3)



<p id="3"></p>



### &sect;3.预处理和特征工程

数据挖掘五大流程：获取数据 —》数据预处理 —》特征工程 —》建模 —》上线验证

**一、数据预处理**

1、数据无量纲化

​	将不同规格的数据转换到同一规格，或不同分布的数据转换到某个特定分布的需求，这种需求统称为将数据“无量纲化”。

​	（1）preprocessing.MinMaxScaler

​	 当数据(x)按照最小值中心化后，再按极差（最大值 - 最小值）缩放，数据移动了最小值个单位，并且会被收敛到

[0,1]之间，而这个过程，就叫做数据归一化。

​	`data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]`

​	`scaler = MinMaxScaler() #实例化`

​	`result_ = scaler.fit_transform(data) #训练和导出结果一步达成`

​	`scaler.inverse_transform(result) #将归一化后的结果逆转`

​	（2）preprocessing.StandardScaler

​	 当数据(x)按均值(μ)中心化后，再按标准差(σ)缩放，数据就会服从为均值为0，方差为1的正态分布（即标准正态分

布），而这个过程，就叫做数据标准化

​	`scaler = StandardScaler() #实例化`

​	`scaler.fit_transform(data) #使用fit_transform(data)一步达成结果`

​	`scaler.inverse_transform(x_std) #使用inverse_transform逆转标准化`

大多数机器学习算法中，会选择**StandardScaler**来进行特征缩放，因为MinMaxScaler对异常值非常敏感。

**二、缺失值：impute.SimpleImputer**

`#填补年龄
Age = data.loc[:,"Age"].values.reshape(-1,1) #sklearn当中特征矩阵必须是二维`	

`from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer() #实例化，默认均值填补
imp_median = SimpleImputer(strategy="median") #用中位数填补
imp_0 = SimpleImputer(strategy="constant",fill_value=0) #用0填补`

`imp_mean = imp_mean.fit_transform(Age) #fit_transform一步完成调取结果
imp_median = imp_median.fit_transform(Age)
imp_0 = imp_0.fit_transform(Age)`

输入“mean”使用均值填补(默认)（仅对数值型特征可用）

输入“median"用中值填补（仅对数值型特征可用）

输入"most_frequent”用众数填补（对数值型和字符型特征都可用）

输入“constant"表示请参考参数“fifill_value"中的值（对数值型和字符型特征都可用）

**三、编码与哑变量**

为了让数据适应算法和库，我们必须将数据进行编码，即是说，将文字型数据转换为数值型。

1、**preprocessing.LabelEncoder**：标签专用，能够将分类转换为分类数值

`from sklearn.preprocessing import LabelEncoder`

`data.iloc[:,-1] = LabelEncoder().fit_transform(data.iloc[:,-1])`

2、**preprocessing.OrdinalEncoder**：特征专用，能够将分类特征转换为分类数值

`from sklearn.preprocessing import OrdinalEncoder`

`data_.iloc[:,1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:,1:-1])`

3、**preprocessing.OneHotEncoder**：独热编码，创建哑变量

​	算法会把舱门，学历这样的分类特征，都误会成是体重这样的分类特征。这是说，我们把分类转换成数字的时候，忽略了数字中自带的数学性质，所以给算法传达了一些不准确的信息，而这会影响我们的建模。







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

​	![hadoop](https://github.com/Vincy-w/SKLearn/raw/master/pic/泛化误差.png)

​	①模型太复杂或者太简单，都会让泛化误差高，我们追求的是位于中间的平衡点

​	②模型太复杂就会过拟合，模型太简单就会欠拟合

​	③对树模型和树的集成模型来说，树的深度越深，枝叶越多，模型越复杂

​	④树模型和树的集成模型的目标，都是减少模型复杂度，把模型往图像的左边移动

2、参数重要性

|       参数        |               对模型在未知数据上的评估性能影响               |  影响程度  |
| :---------------: | :----------------------------------------------------------: | :--------: |
|   n_estimators    |      提升至平稳，n_estimators↑，不影响单个模型的复杂度       |    ⭐⭐⭐⭐    |
|     max_depth     | 有增有减，默认最大深度，即最高复杂度，向复杂度降低的方向调参。max_depth↓，模型更简单，且向图像的左边移动 |    ⭐⭐⭐     |
| min_samples_leaf  | 有增有减，默认最小限制1，即最高复杂度，向复杂度降低的方向调参。min_samples_leaf↑，模型更简单，且向图像的左边移动 |     ⭐⭐     |
| min_samples_split | 有增有减，默认最小限制2，即最高复杂度，向复杂度降低的方向调参。min_samples_split↑，模型更简单，且向图像的左边移动 |     ⭐⭐     |
|   max_features    | 有增有减，默认auto，是特征总数的开平方，位于中间复杂度，既可以向复杂度升高的方向，也可以向复杂度降低的方向调参。max_features↓，模型更简单，图像左移。max_features↑，模型更复杂，图像右移。max_features是唯一的，既能够让模型更简单，也能够让模型更复杂的参数，所以在调整这个参数的时候，需要考虑我们调参的方向 |     ⭐      |
|     criterion     |                    有增有减，一般使用gini                    | 看具体情况 |



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
