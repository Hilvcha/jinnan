# coding : utf-8
import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime
import time

import os
import sys

import xgboost as xgb
from functools import wraps

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{}:{} seconds'.format(func.__module__, func.__name__, round(end - start, 2)))
        return r
    return wrapper

def encode_onehot(df,column_name):
    feature_df=pd.get_dummies(df[column_name],prefix=column_name)
    all = pd.concat([df.drop([column_name],axis=1),feature_df],axis=1)
    return all

def encode_count(df,column_name):
    lbl=preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name]=lbl.transform(list(df[column_name].values))
    return df

def merge_count(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_nunique(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_median(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_sum(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_max(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_min(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_std(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def feat_count(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_nunique(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_std(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_median(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_max(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_min(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_sum(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_var(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_quantile(df, df_feature, fe,value,n,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].quantile(n)).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_quantile" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_skew(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].skew()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_skew" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def action_feats(df, df_features,fe="userid"):
    a = pd.get_dummies(df_features, columns=['actionType']).groupby(fe).sum()
    a = a[[i for i in a.columns if 'actionType' in i]].reset_index()
    df = df.merge(a, on=fe, how='left')
    return df


# --------------------------------------------------------------------------------------------------
base_path=os.path.abspath(os.path.join(__file__,'..'))
train_path=os.path.join(base_path,"jinnan_round1_train_20181227.csv")
test_path=os.path.join(base_path,"jinnan_round1_testA_20181227.csv")

@timethis
def read_data(train_path,test_path):
    train = pd.read_csv(train_path,encoding='gb2312')
    test = pd.read_csv(test_path,encoding='gb2312')
    return train,test


# ---------------------------------------------------------------------------------
@timethis
def clean_data(df):
    df.drop(['B3','B13','A13','A18','A23'],axis=1,inplace=True)

    less_cols=list(df.columns)
    for col in df.columns:
        rate=df.ix[:,col].value_counts(normalize=True,dropna=False).values[0]
        if rate>0.9:
            less_cols.remove(col)
            print(col,rate)

    df=df[less_cols]
    return df

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


def get_labelEncoder(df,c):
    le=preprocessing.LabelEncoder()
    le.fit(df[c].fillna('0'))
    df[c]=le.transform(df[c].fillna('0'))


@timethis
def feature(df):

    cats=[]
    for col in df.columns:
        if df[col].dtype != "object":
            print(col+'是数值')
            cats.append(col)
        else:
            print(col+'转换中')
            if col=='A25':
                df['A25']= df['A25'].map(lambda x:  np.nan if x=='1900/3/10 0:00' else float(x))
                df['A25']=df['A25'].astype(float)
                print(df.dtypes)
                continue
            get_labelEncoder(df,col)

    # A6--------------------------------------
    columns=list(df.columns)
    columns=['A6','A8','A10','A12','A15','A17','A19','A21','A22','A25','A27',
             'B1','B6','B8','B12','B14',]
    df=df[columns]
    for x in range(0,len(columns)):
        for y in range(0,len(columns)) :
            if y > x:
                print('操作'+columns[x]+'+'+columns[y])
                df = merge_median(df,[columns[x]], columns[y], '{}_{}_median'.format(columns[x], columns[y]))
                df = merge_mean(df, [columns[x]], columns[y], '{}_{}_mean'.format(columns[x], columns[y]))
                df = merge_sum(df, [columns[x]], columns[y], '{}_{}_sum'.format(columns[x], columns[y]))
                df = merge_max(df, [columns[x]], columns[y], '{}_{}_max'.format(columns[x], columns[y]))
                df = merge_min(df, [columns[x]], columns[y], '{}_{}_min'.format(columns[x], columns[y]))
                df = merge_std(df, [columns[x]], columns[y], '{}_{}_std'.format(columns[x], columns[y]))

    # # A6--------------------------------------
    # Cross_feature={
    #
    #                 'A5':'A6',      'A5':'A8',     'A5':'A10',     'A5':'A12',      'A5':'A15',      'A5':'A17',       'A5':'A19',
    #                 'A5':'A21',     'A5':'A22',    'A5':'A27',
    #                 'A5':'B1',      'A5': 'B6',    'A5':'B8',
    #
    #                 'A6':'A8',     'A6':'A10',     'A6':'A12',      'A6':'A15',      'A6':'A17',       'A6':'A19',
    #                 'A6':'A21',     'A6':'A22',    'A6':'A27',
    #                 'A6': 'B1',     'A6': 'B6',    'A6': 'B8',     'A6': 'B12',     'A6':'B14',
    #
    #                 'B14':'B1',     'B14': 'B6',   'B14':'B8',      'B14':'B12',
    #                 }
    # for key in Cross_feature:
    #
    #     df=merge_median(df,[key],Cross_feature[key],'{}_{}_median'.format(key,Cross_feature[key]))
    #     df=merge_mean(df,[key],Cross_feature[key],'{}_{}_mean'.format(key,Cross_feature[key]))
    #     df=merge_sum(df,[key],Cross_feature[key],'{}_{}_sum'.format(key,Cross_feature[key]))
    #     df=merge_max(df,[key],Cross_feature[key],'{}_{}_max'.format(key,Cross_feature[key]))
    #     df=merge_min(df,[key],Cross_feature[key],'{}_{}_min'.format(key,Cross_feature[key]))
    #     df=merge_std(df,[key],Cross_feature[key],'{}_{}_std'.format(key,Cross_feature[key]))
    #
    # Count_feature = {'A5': 'A25', 'A5': 'B4', 'A5': 'B5', 'A5': 'B7', 'A5': 'B9', 'A5': 'B10', 'A5': 'B11',
    #                  'A7':'A5',   'A7':'A25', 'A7':'B4',
    #
    #                  }
    # for key in Count_feature:
    #     df = merge_count(df, [key], Count_feature[key], '{}_{}_count'.format(key, Count_feature[key]))
    #     df = merge_nunique(df, [key], Count_feature[key], '{}_{}_nunique'.format(key, Count_feature[key]))

    # df_feat=df[cats]

    print(df)
    return df


def myMSE(preds,xgbtrain):
    label=xgbtrain.get_label()
    score=mean_squared_error(label,preds)*0.5
    return 'myMSE',score


if __name__ == '__main__':
    label='收率'
    id='样本id'
    train,test=read_data(train_path,test_path)
    test_sample = test.ix[:, :1]
    train_label=train[label]

    train.drop([id,label], axis=1,inplace=True)
    test.drop([id], axis=1,inplace=True)

    train=clean_data(train)
    test=clean_data(test)

    train_length=len(train)
    data=pd.concat([train,test])
    print(train.describe())
    print(test.describe())
    print(data.describe())
    data_feat=feature(data)
    train_feat=data_feat.ix[:train_length-1,:]
    test_feat=data_feat.ix[train_length:,:]


    x_train, x_val, y_train, y_val = train_test_split(train_feat, train_label, test_size=0.2,
                                                      random_state=100)
    print('start running ....')
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    param = {'learning_rate': 0.05,
             'n_estimators': 1000,
             'max_depth': 4,
             'min_child_weight': 6,
             'gamma': 0,
             'subsample': 0.8,
             'colsample_bytree': 0.8,
             'eta': 0.05,
             'silent': 1,
             # 'eval_metric':'rmse',
             }

    num_round = 150
    plst = list(param.items())
    watchlist = [(dval, 'eval'), (dtrain, 'train')]

    bst = xgb.train(plst, dtrain, num_round, watchlist, early_stopping_rounds=10,feval=myMSE)
    res = xgb.cv(param, dtrain, num_round, nfold=5,
                 metrics='rmse', seed=1000,
                 callbacks=[xgb.callback.print_evaluation(show_stdv=True), xgb.callback.early_stop(5)],feval=myMSE)
    print(res)
    dtest = xgb.DMatrix(test_feat)
    Pred = bst.predict(dtest)
    test_sample["Pred"]=Pred
    # test = pd.concat([Pred,Pred])
    test_sample.to_csv('submit_{}.csv'.format(time.strftime("%Y-%m-%d_%H-%M", time.localtime()) ), index=False,header=None)

    import pandas as pd
    import matplotlib.pylab as plt
    feat_imp = pd.Series(bst.get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

