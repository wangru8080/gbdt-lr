# gbdt+lr  
参考Facebook论文：http://quinonero.net/Publications/predicting-clicks-facebook.pdf  
##### 原理：GBDT是一种常用的非线性模型，基于集成学习中boosting的思想，由于GBDT本身可以发现多种有区分性的特征以及特征组合，决策树的路径可以直接作为LR输入特征使用，省去了人工寻找特征、特征组合的步骤。所以可以将GBDT的叶子结点输出，作为LR的输入，如图所示：
![a](https://upload-images.jianshu.io/upload_images/4155986-8a4cb50aefba2877.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/508)
