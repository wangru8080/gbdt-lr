import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import gc
from scipy import sparse

def preProcess():
    path = 'data/'
    print('读取数据...')
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')
    print('读取结束')
    df_train.drop(['id'], axis = 1, inplace = True)
    df_test.drop(['id'], axis = 1, inplace = True)

    df_test['target'] = -1

    data = pd.concat([df_train, df_test])
    return data

def lr_predict(data, bin_feats, cat_feats, con_feats):
    # 连续特征归一化
    print('开始归一化...')
    scaler = MinMaxScaler()
    for col in con_feats:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    print('归一化结束')
    
    # 离散特征one-hot编码
    print('开始one-hot...')
    for col in cat_feats + bin_feats:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')
    train = data[data['target'] != -1]
    target = train.pop('target')
    test = data[data['target'] == -1]
    test.drop(['target'], axis = 1, inplace = True)

    # 划分数据集
    print('划分数据集...')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)
    print('开始训练...')
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    val_auc = roc_auc_score(y_val, lr.predict_proba(x_val)[:, 1])
    print('AUC: ', val_auc)
    print('开始预测...')
    y_pred = lr.predict_proba(test)[:, 1]
    print('写入结果...')
    submission = pd.read_csv('data/sample_submission.csv')
    submission['target'] = y_pred
    submission.to_csv('submission/submission_Lr_auc_%s.csv' % val_auc, index = False)
    print('结束')

def gbdt_predict(data, bin_feats, cat_feats, con_feats):
    # 离散特征one-hot编码
    print('开始one-hot...')
    for col in cat_feats + bin_feats:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    train = data[data['target'] != -1]
    target = train.pop('target')
    test = data[data['target'] == -1]
    test.drop(['target'], axis = 1, inplace = True)
    train = data[data['target'] != -1]
    target = train.pop('target')
    test = data[data['target'] == -1]
    test.drop(['target'], axis = 1, inplace = True)

    # 划分数据集
    print('划分数据集...')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)

    print('开始训练..')
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=100,
                            max_depth = 12,
                            learning_rate=0.01,
                            n_estimators=10000,
                            )

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric = 'auc',
            early_stopping_rounds = 100,
            )

    val_auc = roc_auc_score(y_val, gbm.predict(x_val))
    y_pred = gbm.predict(test)
    print('写入结果...')
    submission = pd.read_csv('data/sample_submission.csv')
    submission['target'] = y_pred
    submission.to_csv('submission/submission_gbdt_auc_%s.csv' % val_auc, index = False)
    print('结束')

def gbdt_lr_predict(data, bin_feats, cat_feats, con_feats):
    # 离散特征one-hot编码
    print('开始one-hot...')
    for col in cat_feats + bin_feats:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    train = data[data['target'] != -1]
    target = train.pop('target')
    test = data[data['target'] == -1]
    test.drop(['target'], axis = 1, inplace = True)
    train = data[data['target'] != -1]
    target = train.pop('target')
    test = data[data['target'] == -1]
    test.drop(['target'], axis = 1, inplace = True)

    # 划分数据集
    print('划分数据集...')
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)

    print('开始训练gbdt..')
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=100,
                            max_depth = 12,
                            learning_rate=0.01,
                            n_estimators=30,
                            )

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric = 'auc',
            # early_stopping_rounds = 100,
            )
    model = gbm.booster_
    print('训练得到叶子数')
    gbdt_feats_train = model.predict(train, pred_leaf = True)
    gbdt_feats_test = model.predict(test, pred_leaf = True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)

    # print('Writing transformed training data')
    # transformed_training_matrix = np.zeros([len(gbdt_feats_train), len(gbdt_feats_train[0]) * gbdt_feats_train.shape[1]],
    #                                    dtype=np.int64)
    # for i in range(0, len(gbdt_feats_train)):
    #     temp = np.arange(len(gbdt_feats_train[0])) * gbdt_feats_train.shape[1] + np.array(gbdt_feats_train[i])
    #     transformed_training_matrix[i][temp] += 1

    # transformed_testing_matrix = np.zeros([len(gbdt_feats_test), len(gbdt_feats_test[0]) * gbdt_feats_test.shape[1]],
    #                                    dtype=np.int64)
    # for i in range(0, len(gbdt_feats_test)):
    #     temp = np.arange(len(gbdt_feats_test[0])) * gbdt_feats_test.shape[1] + np.array(gbdt_feats_test[i])
    #     transformed_testing_matrix[i][temp] += 1

    print('构造新的数据集...')
    train = pd.concat([train, df_train_gbdt_feats], axis = 1)
    test = pd.concat([test, df_test_gbdt_feats], axis = 1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    # 连续特征归一化
    print('开始归一化...')
    scaler = MinMaxScaler()
    for col in con_feats:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    print('归一化结束')

    # 叶子数one-hot
    print('开始one-hot...')
    for col in gbdt_feats_name:
        print('this is feature:', col)
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)
    print('one-hot结束')

    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()

    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)
    # lr
    print('开始训练lr..')
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_auc = roc_auc_score(y_train, lr.predict_proba(x_train)[:, 1])
    print('train-AUC: ', tr_auc)
    val_auc = roc_auc_score(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-AUC: ', val_auc)
    print('开始预测...')
    y_pred = lr.predict_proba(test)[:, 1]
    print('写入结果...')
    submission = pd.read_csv('data/sample_submission.csv')
    submission['target'] = y_pred
    submission.to_csv('submission/submission_gbdt+Lr_auc_%s.csv' % val_auc, index = False)
    print('结束')

if __name__ == '__main__':
    data = preProcess()
    bin_feats = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']
    cat_feats = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat']
    con_feats = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']
    # lr_predict(data, bin_feats, cat_feats, con_feats)
    # gbdt_predict(data, bin_feats, cat_feats, con_feats)
    gbdt_lr_predict(data, bin_feats, cat_feats, con_feats)
