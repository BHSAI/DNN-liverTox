# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:47:35 2018

@author: hwang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.externals import joblib
import os

seed = 7
np.random.seed(seed)


def Metrics(y_exp,y_pred):
    m = {}
    TN, FP, FN, TP = metrics.confusion_matrix(y_exp, y_pred).ravel()
    MCC = metrics.matthews_corrcoef(y_exp, y_pred)
    Sen = metrics.recall_score(y_exp, y_pred)
    Spe = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    BAc = 0.5 * (Sen + Spe)
    f1 = metrics.f1_score(y_exp, y_pred)
    
    m['TN']   =TN
    m['FP']   =FP
    m['FN']   =FN
    m['TP']   =TP
    m['MCC']  =MCC
    m['Sen']  =Sen
    m['Spe']  =Spe
    m['PPV']  =PPV
    m['NPV']  =NPV
    m['BAc']  =BAc
    m['f1']   =f1
    return(m)
    

def intersect(a, b):
    return list(set(a) & set(b))

parameter = pd.DataFrame()

n_inner = 4 #cv for model selection
n_outer = 10 #cv for evaluete model
inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=255)
outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=255)

dataset = ['TMG',]


endpoint = ['Biliary.Hyperplasia', ]

dir_data = 'data_matrix' #all datasets should be here

param = pd.DataFrame()
for gs in dataset:
    path = dir_data 
    for tp in endpoint:
        tr_f = os.path.join(path, gs+'.'+tp+'..cons.tr.csv')
        tr = pd.read_csv(tr_f,index_col=0)
        sampleID_tr = tr.index.values
        
        test2_f = os.path.join(path, gs+'.'+tp+'..cons.Ippolito.csv')
        test2 = pd.read_csv(test2_f,index_col=0)
        sampleID_test2 = test2.index.values

        X_tr = tr.drop('Label', axis = 1)    
        X_tr = X_tr.values
        y_tr = tr['Label']
        y_tr = y_tr.values
        
        X_test2 = test2.drop('Label', axis = 1)
        y_test2 = test2['Label']
        y_test2 = y_test2.values
        
        outcomes_train = pd.DataFrame()
        outcomes_val = pd.DataFrame()
        outcomes_test2 = pd.DataFrame()
        
        param_grid = {
                      "min_impurity_decrease": np.arange(0.0, 0.01, 0.01, 0.001),
                      "n_estimators" : [500,1000,2000,3000],
            }
        

        inner = GridSearchCV(
                              estimator=RandomForestClassifier(
                                      class_weight = 'balanced',
                                      random_state = 255),
                              param_grid=param_grid,
                              cv=inner_cv,
                              verbose=1,
                              scoring = 'f1',
                              )
            
        inner.fit(X_tr, y_tr)

        best_param = inner.best_params_
        
        best_param['inner_score'] = inner.best_score_
        best_param['set'] = gs
        best_param['endpoint'] = tp
        best_param_df = pd.DataFrame([best_param])
        param = pd.concat([param,best_param_df],axis=0)
        best_param_df.to_csv(gs+'.'+tp+'.parameters.csv')

        cv_iter = 0
        for train_index, val_index in outer_cv.split(X_tr,y_tr):
            X_train, X_val = X_tr[train_index], X_tr[val_index]
            y_train, y_val = y_tr[train_index], y_tr[val_index]
            
            sampleID_train = list(sampleID_tr[list(train_index)]) 
            sampleID_val = list(sampleID_tr[list(val_index)])
        
            cv_iter += 1
            outer = RandomForestClassifier(
                    n_estimators          =    best_param['n_estimators'],
                    min_impurity_decrease =    best_param['min_impurity_decrease'],
                    class_weight = 'balanced',
                    random_state = 255,
                )
            
            outer.fit(X_train, y_train) #fit models at outer cv

            model_f = gs+'.'+tp+"."+str(cv_iter)+".pkl"
            joblib.dump(outer, model_f)

            clf = joblib.load(model_f)

            predictions_val = clf.predict(X_val)
            pred_f = gs+'.'+tp+"."+str(cv_iter)+".val.predictions.csv"
            save_preds = pd.DataFrame(
                                  {
                                   'Sample':sampleID_val,#sample names
                                   'y_exp': y_val,#ground truth
                                   'y_pred': predictions_val, #round to binary
                                  })
            save_preds.to_csv(pred_f,encoding='utf-8')
                
            m_val = Metrics(y_val, predictions_val)  
            print('\nval mcc: {}  f1: {}'.format(m_val['MCC'], 
                                                 m_val['f1'],
                                                 ))
            
            outcomes_val = outcomes_val.append({
                    'Set'  :gs,
                    'TN'   :m_val['TN'],
                    'FP'   :m_val['FP'],
                    'FN'   :m_val['FN'],
                    'TP'   :m_val['TP'],
                    'MCC'  :m_val['MCC'],
                    'Sen'  :m_val['Sen'],
                    'Spe'  :m_val['Spe'],
                    'PPV'  :m_val['PPV'],
                    'NPV'  :m_val['NPV'],
                    'BAc'  :m_val['BAc'],
                    'f1'   :m_val['f1'],
                    },ignore_index=True)
            outcomes_val = outcomes_val[['Set', 'MCC', 'TP', 'TN', 'FN', 'FP', 'Sen', 'Spe', 'PPV', 'NPV', 'BAc', 'f1']]
            

            #calculate predictions
            predictions_test2 = clf.predict(X_test2)
            pred_f = gs+'.'+tp+"."+str(cv_iter)+".Ippolito.predictions.csv"
            save_preds = pd.DataFrame(
                                  {
                                   'Sample':sampleID_test2,#sample names
                                   'y_exp': y_test2,#ground truth
                                   'y_pred': predictions_test2, #round to binary
                               })
            save_preds.to_csv(pred_f,encoding='utf-8')

            m_test2 = Metrics(y_test2, predictions_test2)
            print('\nIppolito mcc: {}  f1: {}'.format(m_test2['MCC'], 
                                                     m_test2['f1'], 
                                                     ))
            outcomes_test2 = outcomes_test2.append({
                        'Set'  :gs,
                        'TN'   :m_test2['TN'],
                        'FP'   :m_test2['FP'],
                        'FN'   :m_test2['FN'],
                        'TP'   :m_test2['TP'],
                        'MCC'  :m_test2['MCC'],
                        'Sen'  :m_test2['Sen'],
                        'Spe'  :m_test2['Spe'],
                        'PPV'  :m_test2['PPV'],
                        'NPV'  :m_test2['NPV'],
                        'BAc'  :m_test2['BAc'],
                        'f1'   :m_test2['f1'],
                        },ignore_index=True)
            outcomes_test2 = outcomes_test2[['Set', 'MCC', 'TP', 'TN', 'FN', 'FP', 'Sen', 'Spe', 'PPV', 'NPV', 'BAc', 'f1']]
                
        outcomes_val.to_csv(gs+'.'+tp+".metrics_val.csv", encoding='utf-8')
        outcomes_test2.to_csv(gs+'.'+tp+".metrics_Ippolito.csv", encoding='utf-8')
param.to_csv('best_param.csv')
