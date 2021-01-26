from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import datetime
import warnings
warnings.filterwarnings(action='ignore')



# 필요한 함수 정의
def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x     = str(x)
    year  = int(x[:4])
    month = int(x[4:6])
    day   = int(x[6:8])
    hour  = int(x[8:10])
    #min  = int(x[10:12])
    #sec  = int(x[12:])
    return dt.datetime(year, month, day, hour)#, min, sec)

def string2num(x):
    # (,)( )과 같은 불필요한 데이터 정제
    x = re.sub(r"[^0-9]+", '', str(x))
    if x =='':
        return 0
    else:
        return int(x)

PATH = "./"
train_err = pd.read_csv(PATH+'/train_err_data.csv')
train_problem = pd.read_csv(PATH+'/train_problem_data.csv')
train_quality = pd.read_csv(PATH+'/train_quality_data.csv')


def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x     = str(x)
    # print(x)
    year  = int(x[:4])
    month = int(x[4:6])
    day   = int(x[6:8])
    hour  = int(x[8:10])
    #mim  = int(x[10:12])
    #sec  = int(x[12:])
    return dt.datetime(year, month, day, hour)

train_err['is_quality'] = train_err['user_id'].isin(train_quality['user_id'])
train_err['time_until_hour']  = train_err['time'].map(lambda x : make_datetime(x))

train_quality['time'] = pd.to_datetime(train_quality['time'], format="%Y%m%d%H%M%S")


one_hour_before_quality = lambda x : x - dt.timedelta(hours=1)
train_quality['1h_before']=one_hour_before_quality(train_quality['time'])

one_hour_after_quality = lambda x : x + dt.timedelta(hours=1)
train_quality['1h_after']=one_hour_after_quality(train_quality['time'])

one_hour_after_quality = lambda x : x + dt.timedelta(hours=2)
train_quality['2h_after']=one_hour_after_quality(train_quality['time'])

have_quality_err = train_err[train_err.is_quality == 1]
have_qual_err = have_quality_err.loc[:,['user_id', 'time', 'fwver', 'errtype', 'errcode','time_until_hour']].values #model_nm과 is_quality빼고

have_qual_err_list = np.zeros((0,6))



have_qual_err_df = pd.DataFrame(have_qual_err,columns=['user_id','time','fwver','errtype','errcode','time_hr'])
train_quality = train_quality.iloc[:,[1,2,16,17,18]]

def sooyeon(k):
    qt = train_quality[train_quality.user_id < k]
    err = have_qual_err_df[have_qual_err_df.user_id < k]

    tem = pd.merge(err,qt,how='inner',on=['user_id','fwver'])
    tem_nodu = tem[tem.duplicated()==False]

    return tem_nodu[(tem_nodu.time_hr>=tem_nodu['1h_before'])&(tem_nodu.time_hr<=tem_nodu['2h_after'])] 

res1 = sooyeon(11000)
res2 = sooyeon(12000)
res3 = sooyeon(13000)
res4 = sooyeon(14000)
res5 = sooyeon(15000)
res6 = sooyeon(16000)
res7 = sooyeon(17000)
res8 = sooyeon(18000)
res9 = sooyeon(19000)
res10 = sooyeon(20000)
res11 = sooyeon(21000)
res12 = sooyeon(22000)
res13 = sooyeon(23000)
res14 = sooyeon(24000)
res15 = sooyeon(25000)


result = pd.concat([res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12,res13,res14,res15],axis=0)
result.to_csv("have_qual_err_2h.csv", index = True)