---
layout:     post
title:      银行营销数据集（Bank Marketing Data）实验分析
subtitle:   分析报告
date:       2020-10-06
author:     ZHMIYO
header-img: img/post-bg-timg2.jpg
catalog: true
tags:
    - BLOG
    - 模式识别
---

[TOC]



## 背景概述

​		本次实验是选择的是银行营销数据集（Bank Marketing Data）的数据进行实验分析，实验数据集来自于UCI机器学习库，这些数据与葡萄牙银行机构的营销活动相关。这些营销活动以电话沟通为基础，通常情况下，银行的客服人员需要联系客户至少一次，以此确认客户是否将认购该银行的产品（定期存款）。

​		通过分析与葡萄牙银行机构的营销活动（电话）有关的一些数据，预测客户是否会认购定期存款（变量Y）。对此数据集的研究分析对实际工作生产具有十分重要的作用，可以通过预测结果对未来的工作进行一个初步规划，同时也可以对某些用户是否会订阅定期存款提供一个参照。同时，也希望通过对此数据集的分析，能够对模式识别的相关知识有更好的认识，掌握相关使用技巧。

## 数据集说明及思路分析

#### 1、数据集说明

​		首先，本次银行营销数据集（Bank Marketing Data Set）有四个，分别是是bank-additional-full.csv、bank-additional.csv、bank-full.csv、bank.csv。带-full的文件与不带-full区别主要是数据量是否包含完整，完整的数据有41188个例子，而后者只有4119个（提供了最小的数据集来测试更需要计算能力的机器学习算法）。另外，后两个文件（bank-full.csv、bank.csv）是老的版本，输入特征是17个，而新版本是20个输入，他们都有一个输出量y。

​		为了对整个数据集有个初步的认识，首先介绍此数据集中的属性信息（特征），以新版本为例，一共有20个输入变量和1个输出变量。我们又可以把它们分为客户信息、预测相关属性、社会和经济背景属性和输出量。具体信息如下：

##### （1）客户信息

​		Age：年龄；
​		Job：工作，工作类型（分类：行政管理、蓝领、企业家、女佣、管理、 退休、个体户、服务、学生、技术员、失业、未知）
​		Marital：婚姻，婚姻状况（分类：离婚、已婚、单身、未知）（注：“离婚”指离婚或丧偶）
​		Education：教育（分类：基本.4y、Basy.6y、Basy.9y、高中、文盲、专科、大学学位、未知）
​		Default：违约，信用违约吗？（分类：不、是、不知道）
​		Housing：房，有住房贷款吗？（分类：不、是、不知道）
​		Loan：贷款，有个人贷款吗？（（分类：不、是、不知道）

##### （2）预测相关的其他属性

​		Contact：接触方式（分类：移动电话、固定电话）
​		Month：月，最后一个联系月份（分类：MAR、…、NOV、DEC）
​		Day_of_week：每周的天数，最后一周的联系日（分类：Mon、Tue、Wed、Thu、Fri）
​		Duration：持续时间，最后的接触持续时间，以秒为单位
​		Campaign：在此活动期间为该客户端执行的联系人数量(数字，包括最后的联系人)
​		Pdays：上次活动联系客户后经过的天数（数字；999表示以前没有联系过客户）
​		Previous：本次活动之前和本客户的联系次数（数字）
​		Proutcome：前一次营销活动的结果（分类：失败、不存在、成功）

##### （3）社会和经济背景属性

​		Emp.var.rate：就业变化率-季度指标（数字）
​		cons.price.idx：消费者价格指数-月度指标（数字）
​		cons.conf.idx：消费者信心指数-月度指标（数字）
​		euribor3m:：欧元同业拆借利率3个月利率-每日指标（数字）
​		nr.employed：员工人数-季度指标（数字）

##### （4）输出变量：

​		y -客户是否会定期存款？（是、否）

#### 2、思路分析

​	本次实验是一个分类问题，常用的分类算法有KNN、SVM、决策树、朴素贝叶斯、随机森林等，可以使用这些方法对数据进行学习和分类。
​	除了关注算法以外，还要注意数据本身的问题，如上面已经说明的数据集的简单说明，可以看出数据量大且属性较多，那么我们需要对数据进行数据清洗，剔除脏数据，补全缺失的数据等。还需要对数据进行数据预处理，有缩放、归一化等变换方法，降低数据本身对算法产生的影响。
​	整个过程可以总结为，得到数据—>分析问题—>数据预处理—>提取特征—>分类学习—>检验结果。

## 实验过程
#### 1、前期准备：

​		下载数据集；
​		软件环境安装，Python3.7及相关工具库的安装，此次使用的Jupyter Lab。

#### 2、导入基本库和数据

![image-20201006131816700](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006131816700.png)![image-20201006131833099](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006131833099.png)

#### 3、数据探究

​		数据集的列是20个输入和1个输出量的label。

  ![image-20201006131827892](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006131827892.png)

![image-20201006132008541](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132008541.png)

![image-20201006132013695](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132013695.png)

![image-20201006132019521](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132019521.png)		

​		数据集的输出量数据统计，no:36548,yes:4640，数据并不平衡。

![image-20201006132033850](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132033850.png)

​		数据集由41188条例子，共有21个属性。现在分别取其中几个属性以图形化示例看一下与订购情况的关系。
​		职业与认购情况的关系：

​	![image-20201006132135265](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132135265.png)

​		婚姻状况与认购情况的关系：

![image-20201006132148797](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132148797.png)

​		教育背景与认购情况的关系：

![image-20201006132156667](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132156667.png)

​		联系方式与认购情况的关系：

![image-20201006132202534](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132202534.png)





​		月份与认购情况的关系：

![image-20201006132221671](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132221671.png)

![image-20201006132227913](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132227913.png)

#### 4、数据集分割

​		将数据集按照留出法划分，其中70%为训练集，30%作为验证集。

 ![image-20201006132235477](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132235477.png)

![image-20201006132240049](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132240049.png)

![image-20201006132244861](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132244861.png)

#### 5、运行算法

​		本实验调用了10种算法，对数据进行学习，分别是 KNN,、SVC,、Logistic Regression、决策树、XGBboost、随机森林、AdaBoost、GradientBoost、Bagging。具体实现如下：

![image-20201006132300165](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132300165.png)

训练结果：

![image-20201006132306091](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132306091.png)

评定结果的准确率和召回率：

![image-20201006132310852](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132310852.png)

#### 6、数据优化

​		上面的正负样例悬殊过大，即使模型中有平衡正负样例的参数，训练结果中的准确率和召回率也不高。从前文中数据探究里可发现，数据集包括41188个样本，其中负样本为36548,而正样本只有4640。我们可以从反例中随机选取4640个与正例组成衣一个新的数据集进行训练。

![image-20201006132353385](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132353385.png)

​		正负样例达到平衡，再次进行训练可得结果如下：

![image-20201006132400177](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132400177.png)

​		测试结果显示，准确率和召回率都明显提升。说明了正负样例的平衡对分类模型的效果有很大的影响。

#### 7、其他

##### （1）避免过拟合

​	过拟合的介绍这里不做赘述，主要是指过分泛化的能力。较好的解决过拟合的方法是使用交叉验证。

![image-20201006132411444](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132411444.png)

##### （2）分类结果混淆矩阵

![image-20201006132416491](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132416491.png)

![image-20201006132423612](C:\Users\zdy\AppData\Roaming\Typora\typora-user-images\image-20201006132423612.png)

## 实验小结
​		从以上数据分析：
​		1)营销活动月份:我们看到营销活动最高水平的月份是5月份。然而，这个月潜在客户往往拒绝定期存款。对于下一次营销活动，银行最好将营销活动的重点放在3月、9月、10月和12月。
​		2)年龄类别:银行的下一个营销活动应该针对20多岁或更年轻、60多岁或更年长的潜在客户。最年轻的一组有60%的机会存定期存款，而最年长的一组有76%的机会存定期存款。
​		3)职业:不出意外，学生或退休的潜在客户最有可能申请定期存款。退休人员倾向于持有更多的定期存款，以便通过支付利息获得一些现金。学生是另一组过去经常使用定期存款的人群。
​		4)在通话过程中设计一个问题单:由于通话时间是与潜在客户是否开立定期存款最积极相关的特征，因此在通话过程中为潜在客户提供一个有趣的问题单，可能会增加通话时间。
​		还有一些因素没有分析完全，当把数据集中的季节特性、手机和座机的习惯以及其他信息汇总得到的所有策略结合起来，并简化下一个营销活动应该针对的市场对象，银行的下一个营销活动很可能会比当前的营销活动更有效。



