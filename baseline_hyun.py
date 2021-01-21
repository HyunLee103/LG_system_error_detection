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


"""
error 데이터로 FE 
"""
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



"""
quality 데이터 FE
"""
train_quality.describe()
train_quality.info()
train_quality.isnull().sum()

train_quality.quality_0.value_counts()
train_quality.quality_1.value_counts()
train_quality.quality_2.value_counts()
train_quality.quality_3.value_counts() # 다 0
train_quality.quality_4.value_counts() # 다 0
train_quality.quality_5.value_counts()
train_quality.quality_6.value_counts()
train_quality.quality_7.value_counts()
train_quality.quality_8.value_counts()
train_quality.quality_9.value_counts()
train_quality.quality_10.value_counts() # 분포 상대적으로 고름
train_quality.quality_11.value_counts()
train_quality.quality_12.value_counts()


# q0,q1,q2,q5에서 0, -1, 1, 2, 나머지로 변수 만들기
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

q_train = mk_qt_feature(train_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],15000,10000)


# y 만들기(불만 O, X)
problem = np.zeros(15000)
problem[train_problem.user_id.unique()-10000] = 1 
problem.shape
train_y = problem

# X 만들기
err_train = mk_err_feature(train_err,15000,10000)
q_train = mk_qt_feature(train_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],15000,10000)
train_x = np.concatenate((err_train,q_train),axis=1)

print(train_x.shape)
print(train_y.shape)




"""
Modeling - LGBM
"""
def f_pr_auc(probas_pred, y_true):
    labels=y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score=auc(r,p) 
    return "pr_auc", score, True
#-------------------------------------------------------------------------------------
models     = []
recalls    = []
precisions = []
auc_scores   = []
threshold = 0.5
# 파라미터 설정
params =      {
                'boosting_type' : 'gbdt',
                'objective'     : 'binary',
                'metric'        : 'auc',
                'seed': 1015
                }
#-------------------------------------------------------------------------------------
# 5 Kfold cross validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in k_fold.split(train_x):

    # split train, validation set
    X = train_x[train_idx]
    y = train_y[train_idx]
    valid_x = train_x[val_idx]
    valid_y = train_y[val_idx]

    d_train= lgb.Dataset(X, y)
    d_val  = lgb.Dataset(valid_x, valid_y)
    
    #run traning
    model = lgb.train(
                        params,
                        train_set       = d_train,
                        num_boost_round = 1000,
                        valid_sets      = d_val,
                        feval           = f_pr_auc,
                        verbose_eval    = 20, 
                        early_stopping_rounds = 50
                       )
    
    # cal valid prediction
    valid_prob = model.predict(valid_x)
    valid_pred = np.where(valid_prob > threshold, 1, 0)
    
    # cal scores
    recall    = recall_score(    valid_y, valid_pred)
    precision = precision_score( valid_y, valid_pred)
    auc_score = roc_auc_score(   valid_y, valid_prob)

    # append scores
    models.append(model)
    recalls.append(recall)
    precisions.append(precision)
    auc_scores.append(auc_score)

    print('==========================================================')

print(np.mean(auc_scores))




"""
Test
"""
test_user_id_max = 44998
test_user_id_min = 30000
test_user_number = 14999

test_x = mk_err_feature(test_err,test_user_number,test_user_id_min)
q_test = mk_qt_feature(test_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],test_user_number,test_user_id_min)
test_x = np.concatenate((test_x,q_test),axis=1)

# 예측
pred_y_lst = []
for model in models:
    pred_y = model.predict(test_x)
    pred_y_lst.append(pred_y.reshape(-1,1))

pred_ensemble = np.mean(pred_y_lst, axis = 0)

# 제출
sample_submssion = pd.read_csv(PATH+'sample_submission.csv')
sample_submssion['problem'] = pred_ensemble.reshape(-1)
sample_submssion.to_csv("submission/base_quality_1.csv", index = False)
sample_submssion





