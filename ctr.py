# _*_coding:utf-8_*_
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler

# 设置输出不省略
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


# 文件读取
def read_csv_file(f, logging=False):
    data = pd.read_csv(f)
    print('读取数据成功！！！')
    if logging:
        # 查看头5行
        print(data.head(5))
        # 列名
        print(data.columns)
        # 表的统计信息
        print(data.describe())
        # 表的概述信息
        print(data.info())
    return data


# 提取app_categories编码的一级类目
def categories_process_first_class(cate):
    cate = str(cate)
    if len(cate) < 1:
            return 0
    else:
        return int(cate[0])


# 提取app_categories编码的二级类目
def categories_process_second_class(cate):
    cate = str(cate)
    if len(cate) < 3:
        return 0
    else:
        return int(cate[1:])


# 年龄处理，切段
def age_process(age):
    age = int(age)
    if age == 0:
        return 0
    elif age < 15:
        return 1
    elif age < 25:
        return 2
    elif age < 40:
        return 3
    elif age < 60:
        return 4
    else:
        return 5


# 提取省份编码
def process_province(hometown):
    hometown = str(hometown)
    if len(hometown) > 1:
        province = int(hometown[0:2])
    else:
        province = 0
    return province


# 提取城市编码
def process_city(hometown):
    hometown = str(hometown)
    if len(hometown) > 2:
        city = int(hometown[2:])
    else:
        city = 0
    return city


# 提取点击时间的第几天
def get_time_day(t):
    t = str(t)
    t = int(t[0:2])
    return t


# 提取时段，并把一天切成4段
def get_time_hour(t):
    t = str(t)
    t = int(t[2:4])
    if t < 6:
        return 0
    elif t < 12:
        return 1
    elif t < 18:
        return 2
    else:
        return 3


# 评估与计算logloss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll


if __name__ == '__main__':
    # 加载数据集及数据预处理
    train_data = read_csv_file('./data/train.csv', logging=False)
    train_data['clickTime_day'] = train_data['clickTime'].apply(get_time_day)
    train_data['clickTime_hour'] = train_data['clickTime'].apply(get_time_hour)
    test_data = read_csv_file('./data/test.csv', False)
    test_data['clickTime_day'] = test_data['clickTime'].apply(get_time_day)
    test_data['clickTime_hour'] = test_data['clickTime'].apply(get_time_hour)
    ad = read_csv_file('./data/ad.csv', logging=False)
    app_categories = read_csv_file('./data/app_categories.csv', logging=False)
    app_categories["app_categories_first_class"] = app_categories['appCategory'].apply(categories_process_first_class)
    app_categories["app_categories_second_class"] = app_categories['appCategory'].apply(categories_process_second_class)
    user = read_csv_file('./data/user.csv', logging=False)
    user['age_process'] = user['age'].apply(age_process)
    user["hometown_province"] = user['hometown'].apply(process_province)
    user["hometown_city"] = user['hometown'].apply(process_city)
    user["residence_province"] = user['residence'].apply(process_province)
    user["residence_city"] = user['residence'].apply(process_city)

    # 合并数据
    train_user = pd.merge(train_data, user, on='userID')
    train_user_ad = pd.merge(train_user, ad, on='creativeID')
    train_user_ad_app = pd.merge(train_user_ad, app_categories, on='appID')

    # 取出想要的特征
    x_user_ad_app = train_user_ad_app.loc[:, ['creativeID', 'userID', 'positionID',
                                              'connectionType', 'telecomsOperator', 'clickTime_day', 'clickTime_hour',
                                              'age', 'gender', 'education',
                                              'marriageStatus', 'haveBaby', 'residence', 'age_process',
                                              'hometown_province', 'hometown_city', 'residence_province',
                                              'residence_city',
                                              'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform',
                                              'app_categories_first_class', 'app_categories_second_class']]
    # 转化为np的数组
    x_user_ad_app = x_user_ad_app.values
    # 转化数组的数据类型
    x_user_ad_app = np.array(x_user_ad_app, dtype='int32')
    # 标签部分
    y_user_ad_app = train_user_ad_app.loc[:, ['label']].values

    # 模型
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split

    feat_labels = np.array(['creativeID', 'userID', 'positionID','connectionType', 'telecomsOperator',
                            'clickTime_day', 'clickTime_hour', 'age', 'gender', 'education',
                            'marriageStatus', 'haveBaby', 'residence', 'age_process',
                            'hometown_province', 'hometown_city', 'residence_province', 'residence_city',
                            'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform',
                            'app_categories_first_class', 'app_categories_second_class'])

    forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

    forest.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0], ))
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    # 画出特征重要度的柱状图
    import matplotlib.pyplot as plt
    for f in range(x_user_ad_app.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    plt.title('Feature Importances')
    plt.bar(range(x_user_ad_app.shape[1]), importances[indices], color='lightblue', align='center')

    plt.xticks(range(x_user_ad_app.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1, x_user_ad_app.shape[1]])
    plt.tight_layout()
    # plt.savefig('./random_forest.png', dpi=300)
    plt.show()

    # 随机森林调参
    from sklearn.model_selection import GridSearchCV

    param_grid = {'n_estimators': [10, 100, 500, 1000],
                  'max_features': [0.6, 0.7, 0.8, 0.9]}

    rf = RandomForestClassifier()
    rfc = GridSearchCV(rf, param_grid, scoring='neg_log_loss', cv=3, n_jobs=2)
    rfc.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0], ))
    print('随机森林结果：')
    print(rfc.best_score_)
    print(rfc.best_params_)

    # Xgboost调参
    # import os
    # import xgboost as xgb
    #
    # os.environ["OMP_NUM_THREADS"] = "8"  # 并行训练
    # rng = np.random.RandomState(4315)
    #
    # import warnings
    # warnings.filterwarnings("ignore")
    # param_grid = {'max_depth': [3, 4, 5, 7, 9],
    #               'n_estimators': [10, 50, 100, 400, 800, 1000, 1200],
    #               'learning_rate': [0.1, 0.2, 0.3],
    #               'gamma': [0, 0.2],
    #               'subsample': [0.8, 1],
    #               'colsample_bylevel': [0.8, 1]}
    # xgb_model = xgb.XGBClassifier()
    # rgs = GridSearchCV(xgb_model, param_grid, n_jobs=-1)
    # rgs.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0], ))
    # # Xgboost 的结果
    # print(rgs.best_score_)
    # print(rgs.best_params_)
