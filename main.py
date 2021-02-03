# from system_error_LG_dacon.util import mk_time_feature
from lightgbm.callback import early_stopping
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pycaret.classification import *
from collections import Counter, defaultdict

from tqdm import tqdm
import gc
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')

from util import f_pr_auc,mk_fwver_feature,mk_qt_feature,mk_err_feature,fill_quality_missing,err_count,qua_count,tfidf,qual_statics,model_ft,qual_change,check_unique,dataset_trans,mk_time_feature,make_datetime,make_datetime_second,err_count_minus,dataset_trans2,make_datetime_day,nun_err

from scipy.stats import skew
from scipy.stats import norm, kurtosis

import eli5
import lightgbm as lgb
import shap
shap.initjs()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
import scikitplot as skplt
from sklearn import preprocessing

test_user_id_max = 44998
test_user_id_min = 30000
test_user_number = 14999

train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000



def main(sub_name,train=True,model='lgb'):
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

    for qual_num in list(map(lambda x: 'quality_'+ x, [str(i) for i in range(13)])):
        train_quality[qual_num] = train_quality[qual_num].apply(lambda x: float(x.replace(",","")) if type(x) == str else x)
        test_quality[qual_num] = test_quality[qual_num].apply(lambda x: float(x.replace(",", "")) if type(x) == str else x)


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

    model_total=check_unique('model_nm',train_err,test_err)
    errtype_total=check_unique('errtype',train_err,test_err)
    fwver_total = check_unique('fwver',train_err,test_err)

    ## FE
    err_train = mk_err_feature(train_err,15000,10000,complainer_48h_errcode_unique_testtrain,no_complainer_48h_errcode_unique_testtrain)
    # q_train = mk_qt_feature(train_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],15000,10000)
    err_fwver_train = mk_fwver_feature(train_err,15000,10000)
    datalist = dataset_trans(train_err,'train',15000,42,10000,fwver_total)
    seoho_train = np.concatenate(tuple(datalist),axis=1)   
    err_train_count = err_count(train_err,15000,'train')
    train_qual_change = qual_change(train_quality, 15000, 10000)
    model_train = model_ft(train_err,15000)
    # nn_train = err_count_minus(train_err, 15000,10000)
    datalist2 = dataset_trans2(train_err,'train',15000,42,10000, fwver_total)
    seo_train2 = np.concatenate(tuple(datalist2),axis=1)

    train_err_time = mk_time_feature(train_err,15000, 10000, err_mode=True)
    train_qual_time = mk_time_feature(train_quality,15000,10000, err_mode=False)
    train_nun = nun_err(train_err,'train')  

    qt_ch_err_ratio = train_qual_change.reshape(-1,1)/err_train_count
    fw_err_ratio = err_fwver_train.reshape(-1,1)/err_train_count
    fw_model_ratio = err_fwver_train.reshape(-1,1)/model_train
    train_qual_stats = qual_statics(train_quality, 15000, 10000)

    train_x = np.concatenate((err_train, train_qual_stats,train_err_time,train_qual_time, err_train_count,seoho_train,train_qual_change.reshape(-1,1),model_train,qt_ch_err_ratio,fw_err_ratio,fw_model_ratio,seo_train2,train_nun), axis=1)

    
    err_test = mk_err_feature(test_err, test_user_number,test_user_id_min,complainer_48h_errcode_unique_testtrain,no_complainer_48h_errcode_unique_testtrain)
    # q_test = mk_qt_feature(test_quality,['quality_0','quality_1','quality_2','quality_5','quality_6','quality_7','quality_8','quality_9','quality_10','quality_11','quality_12'],test_user_number,test_user_id_min)
    err_fwver_test = mk_fwver_feature(test_err, test_user_number,test_user_id_min)
    test_datalist = dataset_trans(test_err,'test',14999,42,30000,fwver_total)
    seoho_test = np.concatenate(tuple(test_datalist),axis=1)
    err_test_count = err_count(test_err,test_user_number,'test')
    test_qual_change = qual_change(test_quality, test_user_number,test_user_id_min)
    model_test = model_ft(test_err,14999)
    # nn_test = err_count_minus(test_err,14999,30000)
    test_qual_stats = qual_statics(test_quality, test_user_number,test_user_id_min)
    test_datalist2 = dataset_trans2(test_err,'test',14999,42,30000, fwver_total)
    seo_test2 = np.concatenate(tuple(test_datalist2),axis=1)

    test_err_time = mk_time_feature(test_err,test_user_number, test_user_id_min, err_mode=True)
    test_qual_time = mk_time_feature(test_quality,test_user_number, test_user_id_min, err_mode=False)
    test_nun = nun_err(test_err,'test')

    qt_ch_err_ratio_t = test_qual_change.reshape(-1,1)/(err_test_count+1)
    fw_err_ratio_t = err_fwver_test.reshape(-1,1)/err_test_count
    fw_model_ratio_t = err_fwver_test.reshape(-1,1)/model_test.reshape(-1,1)

    test_x = np.concatenate((err_test, test_qual_stats,test_err_time,test_qual_time, err_test_count,seoho_test,test_qual_change.reshape(-1,1),model_test.reshape(-1,1),qt_ch_err_ratio_t,fw_err_ratio_t,fw_model_ratio_t,seo_test2,test_nun), axis=1)

    problem = np.zeros(15000)
    problem[train_problem.user_id.unique()-10000] = 1
    train_y = problem

    train_x = pd.DataFrame(train_x)[pd.DataFrame(train_x).columns.difference([389,94,432,194,323])]
    test_x = pd.DataFrame(test_x)[pd.DataFrame(test_x).columns.difference([389,94,432,194,323])]

    print(train_x.shape)
    print(test_x.shape)

        

    ## modeling
    if train:
        if model == 'automl':              
            train = pd.DataFrame(data=train_x)
            train['problem'] = problem
            clf = setup(data = train, target = 'problem', session_id = 123) 
            cat = create_model('catboost')
            tune_cat = tune_model(cat, optimize='AUC')

            gbc_model = create_model('gbc')

            tune_gbc = tune_model(gbc_model, optimize = 'AUC')

            lgm_model = create_model('lightgbm')

            tune_lgm = tune_model(lgm_model, optimize = 'AUC')

            et_model = create_model('et')

            tune_et = tune_model(et_model, optimize = 'AUC')

            rf_model = create_model('rf')

            tune_rf = tune_model(rf_model, optimize = 'AUC')

            blended5 = blend_models(estimator_list= [tune_cat, tune_gbc, tune_lgm, tune_et, tune_rf], method='soft')

            pred_holdout = predict_model(blended5)

            final_model2 = finalize_model(blended5)

            # best_5 = compare_models(sort = 'AUC', n_select = 5)
            # blended = blend_models(estimator_list = best_5, fold = 5, method = 'soft')
            # pred_holdout = predict_model(blended)
            # final_model = finalize_model(blended)
            
            ## test
            test = pd.DataFrame(data=test_x)
            predictions = predict_model(final_model2, data = test)
            
            
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
            sample_submission.to_csv(f"submission/final_2model_ensemble_2.csv", index = False)
        
        


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
            sample_submission['problem'] = pred_ensemble
            if not os.path.exists('submission'):
                os.makedirs(os.path.join('submission'))
            sample_submission.to_csv(f"submission/28_19_24Ensemble.csv", index = False)


if __name__ == '__main__':
    main('test')



