import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pandas._libs.tslibs import delta_to_tick
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

from util import f_pr_auc
from util import mk_err_feature
from util import mk_qt_feature
from util import mk_time_feature
from util import mk_fwver_feature
from util import mk_time_seg_feature

from fill_in_na import fill_quality_missing


test_user_id_max = 44998
test_user_id_min = 30000
test_user_number = 14999

def main(sub_name,duplicate=False,train=True,model='lgb'):
    ## data load 
    PATH = "data/"
    train_err  = pd.read_csv(PATH+'train_err_data.csv')
    train_quality  = pd.read_csv(PATH+'train_quality_data.csv')
    train_problem  = pd.read_csv(PATH+'train_problem_data.csv')
    test_err  = pd.read_csv(PATH+'test_err_data.csv')
    test_quality  = pd.read_csv(PATH+'test_quality_data.csv')
    

    #결측치 채우기
    train_quality = fill_quality_missing(train_err, train_quality)
    test_quality = fill_quality_missing(test_err, test_quality)


    # fwver_count
    err_fwver_train = mk_fwver_feature(train_err,15000,10000)


    #time
    # err_time_train = mk_time_feature(train_err, 15000, 10000)
    # quality_time_train = mk_time_feature(train_quality, 15000, 10000)
    err_time_seg_train = mk_time_seg_feature(train_err, 15000, 10000)
    # quality_time_seg_train = mk_time_seg_feature(train_quality)


    ### errcode를 위한 전처리
       
    train_err['time'] = pd.to_datetime(train_err['time'], format="%Y%m%d%H%M%S")
    train_problem['time'] = pd.to_datetime(train_problem['time'], format="%Y%m%d%H%M%S")


    train_err['is_complain'] = train_err['user_id'].isin(train_problem['user_id'])
    complainer = train_err[train_err['is_complain']==True]
    no_complainer = train_err[train_err['is_complain']==False]

    complainer_48h_before = np.zeros((0,2))
  

    ### 신고시간 기준 24h이내 train_err (complainer_24h_before) 만들기
    for id in train_problem.user_id.unique():
      #print(id)
      for time in train_problem[train_problem.user_id == id ].time:
        time_48h_before_complain = time - dt.timedelta(days=2)
        temp=(complainer[(complainer['user_id'] == id) & (complainer['time'] > time_48h_before_complain) & (complainer['time'] <= time)][['user_id','errcode']])
        complainer_48h_before= np.concatenate([complainer_48h_before, temp])


    complainer_48h_before = pd.DataFrame(complainer_48h_before , columns=['user_id','errcode'] )



    ## 신고자, 비신고자만 가진 errcode set만들기
   
    complainer_48h_errcode_unique = set(complainer_48h_before.errcode.unique()) - set(no_complainer.errcode.unique())
    no_complainer_48h_errcode_unique = set(no_complainer.errcode.unique()) -set(complainer_48h_before.errcode.unique())

   
    # 신고자, 비신고자만 가진  train, test에 모두 있는 errcode set만들기
    complainer_48h_errcode_unique_testtrain = complainer_48h_errcode_unique.intersection(test_err.errcode.unique())
    no_complainer_48h_errcode_unique_testtrain = no_complainer_48h_errcode_unique.intersection(test_err.errcode.unique())


    # FE
    err_train = mk_err_feature(train_err,15000,10000,complainer_48h_errcode_unique_testtrain,no_complainer_48h_errcode_unique_testtrain)

    ## quality 모두 float 형으로 변환
    for qual_num in list(map(lambda x: 'quality_'+ x, [str(i) for i in range(13)])):
        test_quality[qual_num] = test_quality[qual_num].apply(lambda x: float(x.replace(",","")) if type(x) == str else x)
    for qual_num in list(map(lambda x: 'quality_'+ x, [str(i) for i in range(13)])):
        test_quality[qual_num] = test_quality[qual_num].apply(lambda x: float(x.replace(",","")) if type(x) == str else x)
    
    q_train = mk_qt_feature(train_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],15000,10000)
    

    train_x = np.concatenate((err_train, q_train, err_fwver_train, err_time_seg_train), axis=1)
    
    test_x = mk_err_feature(test_err, test_user_number,test_user_id_min)
    q_test = mk_qt_feature(test_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],test_user_number,test_user_id_min)
    err_fwver_test = mk_fwver_feature(test_err, test_user_number,test_user_id_min)
    
    # time
    err_time_test = mk_time_feature(test_err, test_user_number, test_user_id_min)
    quality_time_test = mk_time_feature(test_quality, test_user_number, test_user_id_min)
    err_time_seg_test = mk_time_seg_feature(test_err,test_user_number,test_user_id_min)
    # quality_time_seg_test = mk_time_seg_feature(test_quality,test_user_number,test_user_id_min)
    test_x = np.concatenate((test_x, q_test, err_fwver_test, err_time_seg_test), axis=1)

    problem = np.zeros(15000)
    problem[train_problem.user_id.unique()-10000] = 1
    train_y = problem



    ## modeling
    if train:
        if model == 'automl':
            train = pd.DataFrame(data=train_x)
            train['problem'] = problem
            clf = setup(data = train, target = 'problem', session_id = 123) 
            best_5 = compare_models(sort = 'AUC', n_select = 5)
            blended = blend_models(estimator_list = best_5, fold = 5, method = 'soft')
            pred_holdout = predict_model(blended)
            final_model = finalize_model(blended)
            
            ## test
            test = pd.DataFrame(data=final_test_x)
            predictions = predict_model(final_model, data = test)
            
            
            sample_submission  = pd.read_csv(PATH+"sample_submission.csv")
            x = []
            for i in range(len(predictions['Score'])):
                if predictions['Label'][i] =='1.0':
                    x.append(predictions['Score'][i])
                else:
                    x.append(1-predictions['Score'][i])
                    
            sample_submission['problem']=x
            #sample_submission.head()
            predictions = predict_model(final_model, data = test)
            #sample_submission.to_csv("AutoML12.csv", index = False)
            if not os.path.exists('submission'):
                os.makedirs(os.path.join('submission'))
            sample_submission.to_csv(f"submission/{sub_name}.csv", index = False)
            
            
            
        if model == 'lgb':
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

                #if model == 'lgb':
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


    #if submit:
            # predict
            pred_y_lst = []
            for model in models:
                pred_y = model.predict(test_x)
                pred_y_lst.append(pred_y.reshape(-1,1))
            pred_ensemble = np.mean(pred_y_lst, axis = 0)

            # submit
            sample_submission = pd.read_csv(PATH+'sample_submission.csv')
            sample_submission['problem'] = pred_ensemble.reshape(-1)
            if not os.path.exists('submission'):
                os.makedirs(os.path.join('submission'))
            sample_submission.to_csv(f"submission/{sub_name}.csv", index = False)

