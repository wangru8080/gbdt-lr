# gbdt+lr  
参考Facebook论文：[Practical Lessons from Predicting Clicks on Ads at
Facebook](http://www.herbrich.me/papers/adclicksfacebook.pdf) 
##### 原理：GBDT是一种常用的非线性模型，基于集成学习中boosting的思想，由于GBDT本身可以发现多种有区分性的特征以及特征组合，决策树的路径可以直接作为LR输入特征使用，省去了人工寻找特征、特征组合的步骤。所以可以将GBDT的叶子结点输出，作为LR的输入，如图所示：
![](https://upload-images.jianshu.io/upload_images/4155986-8a4cb50aefba2877.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/508)  

## 数据集  
数据集采用criteo-Display Advertising Challenge比赛的部分数据集，比赛地址： https://www.kaggle.com/c/criteo-display-ad-challenge/data 全部的数据集下载：http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

### lr:
train-logloss:  0.117204349458  
val-logloss:  0.594054488135  

### gbdt:
train-logloss: 0.319873  
val-logloss: 0.521285  

### gbdt+lr:
tr-logloss:  0.261914566568  
val-logloss:  0.502417918338  

##### 综上：gbdt+lr的val-logloss最小。但是gbdt+lr并不是适用于所有的业务数据，当存在高度稀疏特征的时候，线性模型一般会优于非线性模型。  

### FFM：
![](https://github.com/wangru8080/gbdt-lr/blob/master/ffm.png)   

### gbdt+FFM：
![](https://github.com/wangru8080/gbdt-lr/blob/master/gbdt%2Bffm.png) 

#### gbdt+FFM的效果要优于FFM  

FFM见https://github.com/wangru8080/FFMFormat  
其中FFM使用的是libffm库来训练，代码仅给出了构造数据输入的方法（FFMFormat），构造好输入格式后，直接使用libFFM训练即可

为了快速验证实验，实验中所使用的数据集非常小，所以gbdt+FFM的效果在这个数据集下未必会好于gbdt+lr，至于gbdt+FFM与gbdt+lr的效果哪个更好，在以后的学习过程中还需要进行多次试验
