import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pycaret.classification import *

from tqdm import tqdm
import gc
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')

from util import f_pr_auc,mk_fwver_feature,mk_qt_feature,mk_err_feature,fill_quality_missing,err_count,qua_count,qual_change

from scipy.stats import skew
from scipy.stats import norm, kurtosis

test_user_id_max = 44998
test_user_id_min = 30000
test_user_number = 14999

train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000




def main(sub_name,train=True,split=False,model='lgb'):
    ## data load 
    PATH = "data/"
    train_err  = pd.read_csv(PATH+'train_err_data.csv')
    train_quality  = pd.read_csv(PATH+'train_quality_data.csv')
    train_problem  = pd.read_csv(PATH+'train_problem_data.csv')
    test_err  = pd.read_csv(PATH+'test_err_data.csv')
    test_quality  = pd.read_csv(PATH+'test_quality_data.csv')

    train_qt_id = set(train_quality.user_id) # 8281
    train_err_id = set(train_err.user_id)
    train_noqt_id = train_err_id - train_qt_id

    test_qt_id = set(test_quality.user_id) 
    test_err_id = set(test_err.user_id)
    test_noqt_id = test_err_id - test_qt_id

    train_qt_id = sorted(train_qt_id)
    train_noqt_id = sorted(train_noqt_id)
    test_qt_id = sorted(test_qt_id)
    test_noqt_id = sorted(test_noqt_id)


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


    ## FE
    err_train = mk_err_feature(train_err,15000,10000,complainer_48h_errcode_unique_testtrain,no_complainer_48h_errcode_unique_testtrain)
    q_train = mk_qt_feature(train_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],15000,10000)
    err_fwver_train = mk_fwver_feature(train_err,15000,10000)
    # quality_time_seg_train = mk_time_seg_feature(train_quality,15000,10000)
    # err_time_seg_train = mk_time_seg_feature(train_err, 15000, 10000)
    err_train_count = err_count(train_err,15000,'train')
    # qua_train_count = qua_count(train_quality,15000, 10000,train_qt_id, train_noqt_id)
    train_qual_change = qual_change(train_quality, 15000, 10000)

    train_x = np.concatenate((err_train, q_train, err_fwver_train, err_train_count,train_qual_change), axis=1)


    err_test = mk_err_feature(test_err, test_user_number,test_user_id_min,complainer_48h_errcode_unique_testtrain,no_complainer_48h_errcode_unique_testtrain)
    q_test = mk_qt_feature(test_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],test_user_number,test_user_id_min)
    err_fwver_test = mk_fwver_feature(test_err, test_user_number,test_user_id_min)
    # err_time_seg_test = mk_time_seg_feature(test_err,test_user_number,test_user_id_min)
    # quality_time_seg_test = mk_time_seg_feature(test_quality,test_user_number,test_user_id_min)
    err_test_count = err_count(test_err,test_user_number,'test')
    # qua_test_count = qua_count(train_quality,test_user_number,test_user_id_min,test_qt_id, test_noqt_id)
    test_qual_change = qual_change(test_quality, test_user_number,test_user_id_min)

    
    test_x = np.concatenate((err_test, q_test, err_fwver_test, err_test_count,test_qual_change), axis=1)


    problem = np.zeros(15000)
    problem[train_problem.user_id.unique()-10000] = 1
    train_y = problem

    print(train_x.shape)
    print(test_x.shape)

    pd.DataFrame(train_x).to_csv('train_x_83.csv')
    pd.DataFrame(test_x).to_csv('test_x_83.csv')
    
    if split:
        quality_train_user_id = train_quality['user_id'].unique()
        quality_train_user_id = quality_train_user_id-train_user_id_min
        quality_train_user_id
        quality_train_user_id = quality_train_user_id.tolist()


        nonquality_train_user_id = []
        for i in range(train_user_number):
            nonquality_train_user_id.append(i)
        for i in quality_train_user_id:
            nonquality_train_user_id.remove(i)
            
        quality_test_user_id = test_quality['user_id'].unique()
        quality_test_user_id = quality_test_user_id-test_user_id_min
        quality_test_user_id
        quality_test_user_id = quality_test_user_id.tolist()
        
        nonquality_test_user_id = []
        for i in range(test_user_number):
            nonquality_test_user_id.append(i)
        for i in quality_test_user_id:
            nonquality_test_user_id.remove(i)
        
        train_x2 = np.concatenate((err_train, q_train, err_fwver_train), axis=1)
        test_x2 = np.concatenate((test_x, q_test, err_fwver_test), axis=1)
        qua_train_x = train_x[quality_train_user_id]
        nonqua_train_x = train_x2[nonquality_train_user_id]
        print(qua_train_x.shape)
        print(nonqua_train_x.shape)
        
        qua_test_x = test_x[quality_test_user_id]
        nonqua_test_x = test_x2[nonquality_test_user_id]
        
        qua_train_y = train_y[quality_train_user_id]
        nonqua_train_y = train_y[nonquality_train_user_id]
        
    ## modeling
    if train:
        if model == 'automl':
            if split:
                # quality_id
                train = pd.DataFrame(data=qua_train_x)
                train['problem'] = qua_train_y
                clf = setup(data = train, target = 'problem', session_id = 123) 
                best_5 = compare_models(sort = 'AUC', n_select = 5)
                blended = blend_models(estimator_list = best_5, fold = 5, method = 'soft')
                pred_holdout = predict_model(blended)
                final_model = finalize_model(blended)
                
                ## test
                test = pd.DataFrame(data=test_x)
                qua_predictions = predict_model(final_model, data = test)
                
                
                #nonquality_id
                train = pd.DataFrame(data=nonqua_train_x)
                train['problem'] = nonqua_train_y
                clf = setup(data = train, target = 'problem', session_id = 123) 
                best_5_2 = compare_models(sort = 'AUC', n_select = 5)
                blended_2 = blend_models(estimator_list = best_5_2, fold = 5, method = 'soft')
                pred_holdout = predict_model(blended_2)
                final_model = finalize_model(blended_2)
                test_2 = pd.DataFrame(data=nonqua_test_x)
                nonqua_predictions = predict_model(final_model, data = test_2)
                
                sample_submission  = pd.read_csv(PATH+"sample_submission.csv")
                qua_x = []
                for i in range(len(qua_predictions['Score'])):
                    if qua_predictions['Label'][i] =='1.0':
                        qua_x.append(qua_predictions['Score'][i])
                    else:
                        qua_x.append(1-qua_predictions['Score'][i])
                nonqua_x = []
                for i in range(len(nonqua_predictions['Score'])):
                    if nonqua_predictions['Label'][i] =='1.0':
                        nonqua_x.append(nonqua_predictions['Score'][i])
                    else:
                        nonqua_x.append(1-nonqua_predictions['Score'][i])
                        
                final_prediction = [0 for i in range(test_user_number)]
                id = 0
                for i in quality_test_user_id:
                    final_prediction[i] = qua_x[id]
                    id +=1
                
                id = 0
                for i in nonquality_test_user_id:
                    final_prediction[i] = nonqua_x[id]
                    id +=1
                    
                sample_submission['problem'] = final_prediction
                if not os.path.exists('submission'):
                    os.makedirs(os.path.join('submission'))
                sample_submission.to_csv(f"submission/{sub_name}.csv", index = False)
                
    
            train = pd.DataFrame(data=train_x)
            train['problem'] = problem
            clf = setup(data = train, target = 'problem', session_id = 123) 
            best_5 = compare_models(sort = 'AUC', n_select = 5)
            blended = blend_models(estimator_list = best_5, fold = 5, method = 'soft')
            pred_holdout = predict_model(blended)
            final_model = finalize_model(blended)
            
            ## test
            test = pd.DataFrame(data=test_x)
            predictions = predict_model(final_model, data = test)
            
            
            sample_submission  = pd.read_csv(PATH+"sample_submission.csv")
            x = []
            for i in range(len(predictions['Score'])):
                if predictions['Label'][i] =='1.0':
                    x.append(predictions['Score'][i])
                else:
                    x.append(1-predictions['Score'][i])
                    
            sample_submission['problem']=x

            if not os.path.exists('submission'):
                os.makedirs(os.path.join('submission'))
            sample_submission.to_csv(f"submission/30.csv", index = False)
            

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
                                    early_stopping_rounds = 50,
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

            # predict
            pred_y_lst = []
            for model in models:
                pred_y = model.predict(test_x)
                pred_y_lst.append(pred_y.reshape(-1,1))
            pred_ensemble = np.mean(pred_y_lst, axis = 0)

            # submit
            sample_submission = pd.read_csv(PATH+'sample_submission.csv')
            sample_submission['problem'] = tem
            if not os.path.exists('submission'):
                os.makedirs(os.path.join('submission'))
            sample_submission.to_csv(f"submission/28_19_24Ensemble.csv", index = False)


if __name__ == '__main__':
    main('test')




