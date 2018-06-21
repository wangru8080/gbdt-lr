# gbdt+lr  
参考Facebook论文：http://www.herbrich.me/papers/adclicksfacebook.pdf  
##### 原理：GBDT是一种常用的非线性模型，基于集成学习中boosting的思想，由于GBDT本身可以发现多种有区分性的特征以及特征组合，决策树的路径可以直接作为LR输入特征使用，省去了人工寻找特征、特征组合的步骤。所以可以将GBDT的叶子结点输出，作为LR的输入，如图所示：
![a](https://upload-images.jianshu.io/upload_images/4155986-8a4cb50aefba2877.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/508)  

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
