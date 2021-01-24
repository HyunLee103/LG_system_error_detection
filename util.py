import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')


def f_pr_auc(probas_pred, y_true):
    labels=y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score=auc(r,p) 
    return "pr_auc", score, True

def mk_err_feature(df,user_num,user_min):
    # errtype
    id_error = df[['user_id','errtype']].values
    error = np.zeros((user_num,42))

    for person_idx, err in tqdm(id_error):
        error[person_idx - user_min,err - 1] += 1

    # model_nm
    id_model = df[['user_id','model_nm']].values
    model = np.zeros((user_num,9))

    for idx, mol_nm in tqdm(id_model):  
        model[idx-user_min,int(mol_nm[-1])-1] += 1

    # errcode
    # df.errcode.value_counts()[df.errcode.value_counts()>100000].keys()
    errcode_top14 = ['1', '0', 'connection timeout', 'B-A8002', '80', '79', '14', 'active','2', '84', '85', 'standby', 'NFANDROID2','connection fail to establish']
    id_code = df[['user_id','errcode']].values
    code_df = np.zeros((user_num,14))

    for idx, code in tqdm(id_code):
        if code in errcode_top14:
            code_df[idx-user_min,errcode_top14.index(code)] += 1
        else:
            pass

    return np.concatenate((error,model,code_df),axis=1)

    

def mk_qt_feature(df,vars,user_num,user_min):
    q1 = np.zeros((user_num,5))
    q2 = np.zeros((user_num,5))
    for i, var in enumerate(vars):
        id_q = df[['user_id',var]].values
        res = np.zeros((user_num,5))

        for idx, num in tqdm(id_q):
            if num == 0:
                res[int(idx)-user_min,0] += 1
            elif num == -1:
                res[int(idx)-user_min,1] += 1
            elif num == 1:
                res[int(idx)-user_min,2] += 1
            elif num == 2:
                res[int(idx)-user_min,3] += 1
            else:
                res[int(idx)-user_min,4] += 1

        # 0,1,2,6,8,11,12 거의 비슷, 5,7,9,10 거의 비슷, 각각 평균 내서 사용
        if i in [0,1,2,4,6,9,10]:
            q1 += res
        else:
            q2 += res
    
    return np.concatenate((q1/7,q2/4),axis=1)


def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x = str(x)
    # print(x)
    year = int(x[:4])
    month = int(x[4:6])
    day = int(x[6:8])
    hour = int(x[8:10])
    # min  = int(x[10:12])
    # sec  = int(x[12:])
    return dt.datetime(year, month, day, hour)


def mk_time_feature(df, user_num, user_min):
    # column return 추가 필요

    df['time'] = df['time'].map(lambda x: make_datetime(x))

    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    # hour
    hour_error = df[['user_id', 'hour']].values
    hour = np.zeros((user_num, 24))

    for person_idx, hr in tqdm(hour_error):
        hour[person_idx - user_min, hr - 1] += 1

    # day
    day_error = df[['user_id', 'dayofweek']].values
    day = np.zeros((user_num, 7))

    for person_idx, d in tqdm(day_error):
        day[person_idx - user_min, d - 1] += 1

    return np.concatenate((hour, day), axis=1)

## fwver_count
def mk_fwver_feature(df,user_num,user_min):
    df = df.groupby(['user_id', 'model_nm'])
    user_id_fwver_count = df['fwver'].describe()
    fwver_array = np.array(user_id_fwver_count.unique)
    fwver_count = np.zeros((user_num, 1))
    
    id = 0
    for user_id, model_nm in tqdm(user_id_fwver_count.index):
        fwver_count[user_id-user_min,0] += fwver_array[id]
        id +=1
        
    return fwver_count

def mk_time_seg_feature(df, user_num, user_min):
    # column return 추가 필요

    df['time'] = df['time'].map(lambda x: make_datetime(x))

    df["hour"] = df["time"].dt.hour
    conditionlist = [
        (df['hour'] >= 11) & (df['hour'] < 14),
        (df['hour'] >= 14) & (df['hour'] < 20),
        (df['hour'] >= 20) & (df['hour'] < 24) | (df['hour'] == 0)]

    choicelist = [0, 1, 2]  # lunch :0, Afternoon:1 , Night : 2, others : 3
    df['hour_segment'] = np.select(conditionlist, choicelist, default=3)

    train_time_err = pd.concat([df['user_id'], df['hour_segment']], axis=1).values

    hour_err = np.zeros((user_num, 4))

    print('hour_Err shape', hour_err.shape)
    print('train_time_err shape', train_time_err.shape)

    for person_idx, hr in tqdm(train_time_err):
        hour_err[person_idx - user_min, hr - 1] += 1

    df["dayofweek"] = df["time"].dt.dayofweek

    conditionlist = [
        (df['dayofweek'] >= 1) & (df['hour'] < 5),
        (df['dayofweek'] >= 5) & (df['hour'] < 7)]

    choicelist = [0, 1]  # weekday : 0 ,weekend: 1, monday:2
    df['day_seg'] = np.select(conditionlist, choicelist, default=2)

    train_day_err = pd.concat([df['user_id'], df['day_seg']], axis=1).values

    day_err = np.zeros((user_num, 3))

    for person_idx, day in tqdm(train_day_err):
        day_err[person_idx - user_min, day - 1] += 1

    return np.concatenate((hour_err, day_err), axis=1)


def mk_errtype_feature(df, user_num, user_min, type):
    df = df[~df['errtype'].isin([36])][['user_id', 'errtype']]
    error_type = df[['user_id', 'errtype']].values
    error_type_vl = np.zeros((user_num, 40))

    for person_idx, err in tqdm(error_type):
        # 29개 위로는 29가 없기 때문에 1개씩 땡기고  36위로는 36이 없기 때문에 2개씩 땡겨야한다
        if err > 36:
            err = err - 2
        elif err > 29:
            err = err - 1
        error_type_vl[person_idx - user_min, err - 1] += 1

    # errtype 중 4,5,26이 complainer에서 더 많이 등장함
    error_type_vl_only_complainer = error_type_vl[:, 3] + error_type_vl[:, 4] + error_type_vl[:, 25]

    # errtype 중 23,25이 timeout이 더 많이 등장함
    error_type_vl_timeout = error_type_vl[:, 22] + error_type_vl[:, 24]

    if type == 0:
        # 방법 1 : 총 41 차원 : 39차원 + 2차원(34(com)/4,5,26(no_com))
        return np.concatenate((error_type_vl, error_type_vl_only_complainer.reshape((user_num, 1))), axis=1)

    if type == 1:
        # 방법 2 : 총 42 차원 : 39차원 + 2차원(34(com)/4,5,26(no_com)) + 1차원(23,25(timeout))
        return np.concatenate((error_type_vl, error_type_vl_only_complainer.reshape((user_num, 1)),
                               error_type_vl_timeout.reshape((user_num, 1))), axis=1)