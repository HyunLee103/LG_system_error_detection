import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings(action='ignore')

# 필요한 함수 정의
def make_date(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x     = str(x)
    year  = int(x[:4])
    month = int(x[4:6])
    day   = int(x[6:8])


    return dt.datetime(year, month, day)


def fill_quality_missing(df_err, df_quality):
    df_err['time_day']  = df_err['time'].map(lambda x : make_date(x))
    df_quality['time_day']  = df_err['time'].map(lambda x : make_date(x))

    #fwver 채우기
    for i in len(df_quality[df_quality['fwver'].isna()]):
        df_quality[df_quality['fwver'].isna()][i]['fwver'] =  df_err[(df_err['user_id'] == df_quality[df_quality['fwver'].isna()][i]['user_id']) & (df_err['time_day'] ==df_quality[df_quality['fwver'].isna()][i]['time_day'])]['fwver'][0]


    #quality_n 채우기
    qual_list = ['quality_0', 'quality_1', 'quality_2', 'quality_5', 'quality_6', 'quality_7', 'quality_8', 'quality_9', 'quality_11', 'quality_12']
    for i in qual_list:
        df_quality[df_quality[i].isna()] = 0

    df_quality[df_quality['quality_10'].isna()] = 3
    
    return df_quality
    