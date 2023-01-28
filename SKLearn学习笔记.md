# Scikit-Learn学习笔记

章节0：[环境配置](#0)

章节1：[决策树](#1)

章节2：[随机森林](#2)

章节3：[预处理和特征工程](#3)

章节4：[降维算法](#4)

章节5：[逻辑回归](#5)

章节6：[聚类算法](#6)

章节7：[SVM上(支持向量机)](#7)

章节8：[SVM下](#8)



<p id="8"></p>



### &sect;8.SVM(下)

**一、概念**

1、核函数

​	极端情况下，数据可能会被映射到无限维度的空间中，这种高维空间可能不是那么友好，维度越多，推导和计算的难度都会随之暴增。而解决这些问题的数学方式，叫做“核技巧”(Kernel Trick)，是一种能够使用数据原始空间中的向量计算来表示升维后的空间中的点积结果的数学方式。具体表现为，**K(u,v)=Φ(u)·Φ(v)** 。而这个原始空间中的点积函数 **K(u,v)**，就被叫做“核函数"

**二、重要参数**

1、kernel

​	作为SVC类最重要的参数之一，“kernel"在sklearn中可选以下几种选项:

| 输入      | 含义       | 解决问题 | 参数gamma | 参数degree | 参数coef0 |
| --------- | ---------- | -------- | --------- | ---------- | --------- |
| "linear"  | 线性核     | 线性     | No        | No         | No        |
| "poly"    | 多项式核   | 偏线性   | Yes       | Yes        | Yes       |
| "sigmoid" | 双曲正弦核 | 非线性   | Yes       | No         | Yes       |
| "rbf"     | 高斯径向基 | 偏非线性 | Yes       | No         | No        |

**三、性质探索**

1、Kernel选取

2、degree&gamma&coef0选取

**四、软间隔重要参数C**

在实际使用中，C和核函数的相关参数（gamma，degree等等）们搭配，往往是SVM调参的重点。



<p id="7"></p>



### &sect;7.SVM(上)

**一、概念**

1、超平面

​	在几何中，超平面是一个空间的子空间，它是维度比所在空间小一维的空间。 如果数据空间本身是三维的，则其超平面是二维平面，而如果数据空间本身是二维的，则其超平面是一维的直线。

​	在二分类问题中，如果一个超平面能够将数据划分为两个集合，其中每个集合中包含单独的一个类别，我们就说这个超平面是数据的“决策边界”。

**二、损失函数**

![hadoop](https://github.com/Vincy-w/SKLearn/raw/master/pic/SVM损耗函数.png)

d=2/w，要最大化d，就要求解w的最小值。可以转化为我w^2/2的最小值。（此式即为SVM的损失函数）



<p id="6"></p>



### &sect;6.聚类算法

**一、无监督学习与聚类算法**

​	决策树，随机森林，PCA和逻辑回归，他们虽然有着不同的功能，但却都属于“有监督学习”的一部分，即是说，模型在训练的时候，即需要特征矩阵X，也需要真实标签y。机器学习当中，还有相当一部分算法属于“无监督学习”，无监督的算法在训练的时候只需要特征矩阵X，不需要标签。而聚类算法，就是无监督学习的代表算法。

**二、KMeans**

​	作为聚类算法的典型代表，KMeans可以说是最简单的聚类算法。

1、簇与质心

​	KMeans算法将一组N个样本的特征矩阵X划分为K个无交集的簇，直观上来看是簇是一组一组聚集在一起的数据，在一个簇中的数据就认为是同一类。簇就是聚类的结果表现。

​	簇中所有数据的均值μj通常被称为这个簇的“质心”（centroids）。在一个二维平面中，一簇数据点的质心的横坐标就是这一簇数据点的横坐标的均值，质心的纵坐标就是这一簇数据点的纵坐标的均值。

2、簇内差异——Inertia

​	欧式距离来衡量

**三、重要参数**

1、n_clusters

​	n_clusters是KMeans中的k，表示着我们告诉模型我们要分几类。这是KMeans当中唯一一个必填的参数，默认为8类，但通常我们的聚类结果会是一个小于8的结果。

2、评估指标——轮廓系数

​	1）样本与其自身所在的簇中的其他样本的相似度**a**，等于样本与同一簇中所有其他点之间的平均距离

​	2）样本与其他簇中的样本的相似度**b**，等于样本与下一个最近的簇中得所有点之间的平均距离

​	单个样本的轮廓系数公式为：

![hadoop](https://github.com/Vincy-w/SKLearn/raw/master/pic/轮廓系数.png)

3、基于轮廓系数来选择n_clusters



<p id="5"></p>



### &sect;5.逻辑回归

**一、优点**

1、逻辑回归对线性关系的拟合效果极好

2、对于线性数据，逻辑回归的拟合和计算都非常快，计算效率优于SVM和随机森林，尤其在大型数据上能够看得出区别

3、逻辑回归返回的分类结果不是固定的0，1，而是以小数形式呈现的类概率数字，因此可以把逻辑回归返回的结果当成连续型数据来利用。

**二、正则化**

​	正则化是用来防止模型过拟合的过程，常用的有L1正则化和L2正则化两种选项，分别通过在损失函数后加上参数向量 的L1范式和L2范式的倍数来实现。其中L1范数表现为参数向量中的每个参数的绝对值之和，L2范数表现为参数向量中的每个参数的平方和的开方值。

![hadoop](https://github.com/Vincy-w/SKLearn/raw/master/pic/正则化L1L2.png)

|  参数   |                             说明                             |
| :-----: | :----------------------------------------------------------: |
| penalty |                  选择正则方式L1，L2。默认L2                  |
|    C    | C正则化强度的倒数，须为大于0的浮点数，默认1.0。<br />C越小，对损失函数的惩罚越重，正则化的效力越强，参数θ会逐渐被压缩得越来越小。 |

在实际使用时，基本就默认使用l2正则化，如果感觉到模型的效果不好，那就换L1试试看。



<p id="4"></p>



### &sect;4.降维算法

**一、PCA与SVD**

​	主成分分析(Principal Component Analysis，PCA）使用的信息量衡量指标，就是样本方差Var，又称可解释性方差，方差越大，特征所带的信息量越多。

​	奇异值分解(singular value decomposition,SVD)

**二、重要参数**

​	n_components是我们降维后需要的维度，即降维后需要保留的特征数量，降维流程中第二步里需要确认的k值，一般输入[0, min(X.shape)]范围中的整数。



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

![hadoop](https://github.com/Vincy-w/SKLearn/raw/master/pic/独热编码.png)

`from sklearn.preprocessing import OneHotEncoder
X = data.iloc[:,1:-1]`

`OneHotEncoder(categories='auto').fit_transform(X).toarray()`

`#依然可以还原
pd.DataFrame(enc.inverse_transform(result))`

**四、二值化与分段**

1、**sklearn.preprocessing.Binarizer**

​	根据阈值将数据二值化（将特征值设置为0或1），用于处理连续型变量

2、**preprocessing.KBinsDiscretizer**

​	这是将连续型变量划分为分类变量的类，能够将连续型变量排序后按顺序分箱后编码。总共包含三个重要参数：

|     参数     |                          含义&输入                           |
| :----------: | :----------------------------------------------------------: |
|  **n_bins**  |                  每个特征中分箱个数，默认5                   |
|  **encode**  | 编码方式，默认“onehot”<br />“onehot”:做哑变量，之后返回一个稀疏矩阵，每一列是一个特征中的一个类别，含有该类别的样本表示为1，不含的表示为0<br />"ordinal":每个特征的每个箱都被编码为一个整数，返回每一列是一个特征，每个特征下含有不同整数编码的箱的矩阵<br />"onehot-dense":做哑变量，之后返回一个密集数组。 |
| **strategy** | 用来定义箱宽的方式，默认"quantile" <br /> "uniform"：表示等宽分箱，即每个特征中的每个箱的最大值之间的差为max-min/n_bins<br />"quantile"：表示等位分箱，即每个特征中的每个箱内的样本数量都相同<br />"kmeans"：表示按聚类分箱，每个箱中的值到最近的一维k均值聚类的簇心得距离都相同 |



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
