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
from collections import Counter, defaultdict
warnings.filterwarnings(action='ignore')


def f_pr_auc(probas_pred, y_true):
    labels=y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score=auc(r,p) 
    return "pr_auc", score, True


def mk_err_feature(df,user_num,user_min,complainer_48h_errcode_unique_testtrain,no_complainer_48h_errcode_unique_testtrain):

    df['typecode'] = df.errtype.astype(str) + df.errcode.astype(str)
    id_err_var = df[['user_id','typecode','errtype','fwver','errcode']].values

    # 빈 array 생성
    typecode_arr = np.zeros((user_num,3))
    type_arr = np.zeros((user_num,47))
    fwver_arr = np.zeros((user_num,13))
    code_arr = np.zeros((user_num, 17))

    for idx, typecode,type, fwver, code in tqdm(id_err_var):

        # type + code
        if typecode in ['101','23connection fail to establish']:
            typecode_arr[idx - user_min,0] += 1
        elif typecode in ['40','332','261','141','151','161','111','121']:
            typecode_arr[idx - user_min,1] += 1

        typecode_arr[idx - user_min,2] = (typecode_arr[idx - user_min,0]+1)/(typecode_arr[idx - user_min,1]+1)
        
        # errtype
        type_arr[idx - user_min,type - 1] += 1

        if type in [25,18,20,19,21]:
            type_arr[idx - user_min,42] += 1
        elif type in [34,10,35,13,30,27,28]:
            type_arr[idx - user_min,43] += 1
        elif type in [2,4,42,26]:
            type_arr[idx - user_min,44] += 1
        elif type in [1,8]:
            type_arr[idx - user_min,45] += 1
        type_arr[idx - user_min,46] = (type_arr[idx - user_min,42]+1)/(type_arr[idx - user_min,45]+1)   

        # fwver
        fwver_dict = {'05.15.2138':0,'04.22.1750':1,'04.33.1261':2,'04.16.3553':3,'03.11.1167':4,'04.22.1778':5,'04.22.1684':6,'04.33.1185':7,'04.16.3571':8}
        try:
            fwver_arr[idx-user_min,fwver_dict[fwver]] += 1
        except:
            fwver_arr[idx-user_min,9] += 1

        if fwver in ['04.33.1149','04.73.2571','04.16.3571']:
            fwver_arr[idx-user_min,10] += 1
        elif fwver in ['05.15.2120','10']:
            fwver_arr[idx-user_min,11] += 1
        elif fwver in ['04.73.2237','04.22.1684','05.15.2138']:
            fwver_arr[idx-user_min,12] += 1

        # errcode
        errcode_top14 = ['1', '0', 'connection timeout', 'B-A8002', '80', '79', '14', 'active','2', '84', '85', 'standby', 'NFANDROID2','connection fail to establish']
        if code in errcode_top14:
            code_arr[idx-user_min,errcode_top14.index(code)] += 1
        elif code in list(complainer_48h_errcode_unique_testtrain)+['5','6','V-21008','terminate by peer user']:
            code_arr[idx-user_min,14] += 1
        # elif code in ['H-51042','connection fail to establish','4','14','13','83','3','connection timeout']:
        #     code_arr[idx-user_min,15] += 1
        elif code in list(no_complainer_48h_errcode_unique_testtrain)+['Q-64002','S-65002','0']:
            code_arr[idx-user_min,15] += 1
        code_arr[idx-user_min,16] = (code_arr[idx-user_min,14]+1)/(code_arr[idx-user_min,15]+1)

    # 변수 평균 분산 추가
    type_mean = type_arr[:,42:].mean(axis=1)
    type_std = type_arr[:,42:].std(axis=1)

    typecode_mean = typecode_arr.mean(axis=1)
    typecode_std = typecode_arr.std(axis=1)

    fwver_arr_mean = fwver_arr[:,9:].mean(axis=1)
    fwver_arr_std = fwver_arr[:,9:].std(axis=1)

    code_mean = code_arr[:,:14].mean(axis=1)
    code_std = code_arr[:,:14].std(axis=1)

    mean_var = np.concatenate((type_mean.reshape(-1,1),type_std.reshape(-1,1),typecode_mean.reshape(-1,1),typecode_std.reshape(-1,1),fwver_arr_mean.reshape(-1,1),fwver_arr_std.reshape(-1,1),code_mean.reshape(-1,1),code_std.reshape(-1,1)),axis=1)

    return np.concatenate((typecode_arr,type_arr,fwver_arr,code_arr),axis=1)


def mk_qt_feature(df,vars,user_num,user_min):

    for qual_num in list(map(lambda x: 'quality_'+ x, [str(i) for i in range(13)])):
        df[qual_num] = df[qual_num].apply(lambda x: float(x.replace(",","")) if type(x) == str else x)

    q1 = np.zeros((user_num,6))
    q2 = np.zeros((user_num,6))
    q3 = np.zeros((user_num,1))
    qt_cnt = df.groupby('user_id').count()['time']/12
    dict = {key:value for key,value in zip(qt_cnt.index,qt_cnt.values)}
    for i in range(user_num):
        if i+user_min in dict.keys():
            q3[i,0] = dict[i+user_min]

    # 0,1,2,6,8,11,12 거의 비슷, 5,7,9,10 거의 비슷, 각각 평균 내서 사용
    for i, var in enumerate(vars):
        id_q = df[['user_id',var]].values
        res = np.zeros((user_num,6))

        for idx, num in tqdm(id_q):
            if num == 0:
                res[int(idx)-user_min,0] += 1
            elif num == -1:
                res[int(idx)-user_min,1] += 1
            elif num == 1:
                res[int(idx)-user_min,2] += 1
            elif num == 2:
                res[int(idx)-user_min,3] += 1
            elif num == 3:
                res[int(idx)-user_min,4] += 1
            else:
                res[int(idx)-user_min,5] += 1
        q1 += res

        qt_mean = q1.mean(axis=1)
        qt_var = q1.std(axis=1)

        # q1 = q1/q1.sum(axis=1).shape(-1,1)
        
    return np.concatenate((q1/11,q3,qt_mean.reshape(-1,1),qt_var.reshape(-1,1)),axis=1)


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

    # df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    # # hour
    # hour_error = df[['user_id', 'hour']].values
    # hour = np.zeros((user_num, 24))
    #
    # for person_idx, hr in tqdm(hour_error):
    #     hour[person_idx - user_min, hr - 1] += 1

    # day
    day_error = df[['user_id', 'dayofweek']].values
    day = np.zeros((user_num, 4))

    for person_idx, d in tqdm(day_error):
        if d == 1:
            day[person_idx - user_min, 0] += 1
        if d == 5:
            day[person_idx - user_min, 1] += 1
        if d == 6:
            day[person_idx - user_min, 2] += 1
        else:
            day[person_idx - user_min, 3] += 1

    df_day = pd.DataFrame(day, columns=['Mon', 'Sat', 'Sun', 'others'])
    df_day['all'] = df_day['Mon'] + df_day['Sat'] + df_day['Sun'] + df_day['others']

    for var in ['Mon', 'Sat', 'Sun', 'others']:
        df_day[var + '_pct'] = df_day[var] / df_day['all']

    del df_day['all']

    return df_day.values

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


def make_date(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x     = str(x)
    year  = int(x[:4])
    month = int(x[4:6])
    day   = int(x[6:8])
    return dt.datetime(year, month, day)



def fill_quality_missing(df_err, df_quality):
    # df_err['time_day']  = df_err['time'].map(lambda x : make_date(x))
    # df_quality['time_day']  = df_err['time'].map(lambda x : make_date(x))

    # #fwver 채우기
    # for i in len(df_quality[df_quality['fwver'].isna()]):
    #     df_quality[df_quality['fwver'].isna()][i]['fwver'] =  df_err[(df_err['user_id'] == df_quality[df_quality['fwver'].isna()][i]['user_id']) & (df_err['time_day'] ==df_quality[df_quality['fwver'].isna()][i]['time_day'])]['fwver'][0]


    #quality_n 채우기
    qual_list = ['quality_0', 'quality_1', 'quality_2', 'quality_5', 'quality_6', 'quality_7', 'quality_8', 'quality_9', 'quality_11', 'quality_12']
    for i in qual_list:
        df_quality[i].fillna(0)

    df_quality['qulity_10'].fillna(3)
    
    return df_quality


def err_count(df,user_num, df_cat):
    if df_cat == 'train':
        n_total_train = df.groupby('user_id')['user_id'].count()
        #print(n_total_train.shape)
        output= np.array(n_total_train).reshape(user_num,1)
    else:
        n_total_test = df.groupby('user_id')['user_id'].count()
        total_test_list = n_total_test.tolist()
        total_test_list.insert(13262,0)
        output= np.array(total_test_list).reshape(user_num,1)
        #test_x3.shape
        
    return output


def qua_count(df,user_num, user_min,qt_id, noqt_id):
    qua_count = df.groupby('user_id')['user_id'].count()/12
    qua_count_mean = qua_count.mean()
    qua_count_list = [0 for i in range(user_num)]
    
    id=0
    for i in qt_id:
        i = i-user_min
        qua_count_list[i] = qua_count.iloc[id]
        id+=1
    for i in noqt_id:
        i = i-user_min
        qua_count_list[i] = qua_count_mean
    return np.array(qua_count_list).reshape(user_num,1)