import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')

# data load 
PATH = "data/"
train_err  = pd.read_csv(PATH+'train_err_data.csv')
train_quality  = pd.read_csv(PATH+'train_quality_data.csv')
train_problem  = pd.read_csv(PATH+'train_problem_data.csv')
test_err  = pd.read_csv(PATH+'test_err_data.csv')
test_quality  = pd.read_csv(PATH+'test_quality_data.csv')


"""
EDA 
"""
## train_err
train_err.head()
train_err.shape # 1655만
train_err.user_id.nunique() # user 1.5만
train_err.isnull().sum() # err_code 결측값 1개

# time: 10/31 12시 ~ 12/2 18시
train_err.time.min()
train_err.time.max()

# model_nm: 9개, 0~4번 모델이 99%
train_err.model_nm.value_counts()

# fwver: 37개, 상위 6개가 95%
train_err.fwver.nunique()

# errtype: 1~42, 29를 제외한 41개
train_err.errtype.value_counts()

# errcode: 2805개, cnt 10만개 이상인건(97%) 14개
train_err.errcode.value_counts()[train_err.errcode.value_counts()>100000]

## train_quality
train_quality.head()
train_quality.shape # 82만
train_quality.isnull().sum() # fwver, q0, q2, q5 결측값 존재
train_quality.user_id.nunique() # 8281 user_id, 60% 정도 있음 -> 나머지는??

# time: 10/31 12시 ~ 11/30 23시
train_quality.time.min()
train_quality.time.max()

## train_problem
train_problem.shape # 5429개 불만 접수
train_problem.user_id.nunique() # 5000명 평균 한 번 불만 접수, 나머지 만명은?
train_problem.user_id.value_counts()

# time: 11/01 ~ 11/30 23시
train_problem.time.min() 
train_problem.time.max()

# user_id 비교
train_pr_id = set(train_problem.user_id) # 5000
train_qt_id = set(train_quality.user_id) # 8281
train_err_id = set(train_err.user_id) # 1.5만

# quality 로그 중 불만 접수 된 user_id, 3167명
qt_pr_id = train_pr_id & train_qt_id
# quality 로그 중 불만 접수 없는 user_id, 5114명
qt_nopr_id = train_qt_id - train_pr_id

# err에서 불만 접수 된 user_id, 5천명
err_pr_id = train_err_id & train_pr_id
# err에서 불만 접수 없는 user_id, 만명
err_nopr_id = train_err_id - train_pr_id

## fwver에 따른 qt 분포 확인
train_quality.groupby(['fwver']).mean()
train_quality.fwver.nunique()
train_quality_main = train_quality[train_quality['fwver'].isin(['03.11.1167','04.16.3553','04.22.1750','04.22.1778','04.33.1185','04.33.1261','05.15.2138'])]
train_quality.groupby(['user_id']).sum()

# 결측값 처리
train_quality.isnull().sum()

train_quality['quality_5'] = train_quality['quality_5'].fillna(0)
train_quality['quality_0'] = train_quality['quality_0'].fillna(0)
train_quality.dropna(inplace=True)

train_quality.shape # 78만







