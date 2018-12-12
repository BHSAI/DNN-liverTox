# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:07:42 2018

@author: hwang
representative codes for DNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import ParameterGrid

import os

from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

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

dataset = ['TMG']

endpoint = ['Biliary.Hyperplasia']

dir_data = 'data_matrix' #expression data

param = pd.DataFrame()
for gs in dataset:
    path = dir_data
    for tp in endpoint:
        tr_f = os.path.join(path, gs+'.'+tp+'..cons.tr.csv')
        tr = pd.read_csv(tr_f,index_col=0)
        sampleID_tr = tr.index.values
        
        test_f = os.path.join(path, gs+'.'+tp+'..cons.Ippolito.csv')
        test = pd.read_csv(test_f,index_col=0)
        sampleID_test = test.index.values

        X_tr = tr.drop('Label', axis = 1)    
        X_tr = X_tr.values
        y_tr = tr['Label']
        y_tr = y_tr.values
        
        X_test = test.drop('Label', axis = 1)
        y_test = test['Label']
        y_test = y_test.values
        
        outcomes_val = pd.DataFrame()
        outcomes_test = pd.DataFrame()
        
		  #DNN hyperparameters
        nn = [300,300]
        n_batch = 40
        n_epoch = 1000
        n_dropout = 0.8

        rd = 0
        n_run=10
        skf = StratifiedKFold(n_splits=n_run, random_state=255, shuffle=True)
        
        for train_index, val_index in skf.split(X_tr, y_tr):
            rd=rd+1
            print("split cv {}".format(rd))

            X_train, X_val = X_tr[train_index], X_tr[val_index]
            y_train, y_val = y_tr[train_index], y_tr[val_index]
            
            sampleID_train = list(sampleID_tr[list(train_index)]) 
            sampleID_val = list(sampleID_tr[list(val_index)])
            
            X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)

            input_dim = X_resampled.shape[1]
            model = Sequential()
            model.add(Dense(nn[0], input_dim=input_dim, 
                            kernel_initializer='normal', 
                            activation='relu'))
            model.add(Dropout(n_dropout))
            
            for n_node_ix in range(1,len(nn)):
                model.add(Dense(nn[n_node_ix], 
                            kernel_initializer='normal', 
                            activation='relu'))
                model.add(Dropout(n_dropout))
            
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            
            model_f = gs+'.'+tp+"."+str(rd)+".arch.json"
            model_json = model.to_json()
            with open(model_f, "w") as json_file:
                json_file.write(model_json)

            model.compile(loss='binary_crossentropy', optimizer='adam')
                
            early_stopping = EarlyStopping(
                                           monitor='val_loss', 
                                           patience=100, verbose=0, mode='auto')

            weight_f=gs+'.'+tp+"."+str(rd)+".weights-best.hdf5"
            
            checkpoint = ModelCheckpoint(weight_f, 
                                         monitor='val_loss',
                                         verbose=0, 
                                         save_best_only=True,
                                         mode='min')
                
            callbacks_list = [early_stopping,
                              checkpoint,
                              ]
            
            train_history = model.fit(X_resampled, y_resampled,
                      epochs=n_epoch, batch_size=n_batch,  verbose=0,
                      validation_split=0.1,
                      callbacks = callbacks_list)
            
            json_file = open(model_f, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model_best = model_from_json(loaded_model_json)
            model_best.load_weights(weight_f)
            model_best.compile(loss='binary_crossentropy', optimizer='adam')
            
            pred_val = model_best.predict(X_val)
            pred_round = [round(x[0]) for x in pred_val]
            predictions_val = np.array(pred_round)
                
            m_val = Metrics(y_val, predictions_val)  
            print('\nval mcc: {}  f1: {}'.format(m_val['MCC'], 
                                                 m_val['f1'],
                                                 ))
            
            outcomes_val = outcomes_val.append({
                    'Set'  :gs,
                    'Endpoint':tp,
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
            
            pred_f = gs+'.'+tp+"."+str(rd)+".val.predictions.csv"
            save_preds = pd.DataFrame(
                                  {
                                   'Sample':sampleID_val,#sample names
                                   'y_exp': y_val,#ground truth
                                   'pred_score': pred_val.flatten(), #raw score
                                   'y_pred': predictions_val, #round to binary
                                  })
            save_preds.to_csv(pred_f,encoding='utf-8')

            pred_test = model_best.predict(X_test)
            pred_round = [round(x[0]) for x in pred_test]
            predictions_test = np.array(pred_round)
                
            m_test = Metrics(y_test, predictions_test)  
            print('\nIppolito mcc: {}  f1: {}'.format(m_test['MCC'], 
                                                 m_test['f1'],
                                                 ))
            
            outcomes_test = outcomes_test.append({
                    'Set'  :gs,
                    'Endpoint':tp,
                    'TN'   :m_test['TN'],
                    'FP'   :m_test['FP'],
                    'FN'   :m_test['FN'],
                    'TP'   :m_test['TP'],
                    'MCC'  :m_test['MCC'],
                    'Sen'  :m_test['Sen'],
                    'Spe'  :m_test['Spe'],
                    'PPV'  :m_test['PPV'],
                    'NPV'  :m_test['NPV'],
                    'BAc'  :m_test['BAc'],
                    'f1'   :m_test['f1'],
                    },ignore_index=True)
            outcomes_test = outcomes_test[['Set', 'MCC', 'TP', 'TN', 'FN', 'FP', 'Sen', 'Spe', 'PPV', 'NPV', 'BAc', 'f1']]
            
            pred_f = gs+'.'+tp+"."+str(rd)+".Ippolito.predictions.csv"
            save_preds = pd.DataFrame(
                                  {
                                   'Sample':sampleID_test,#sample names
                                   'y_exp': y_test,#ground truth
                                   'pred_score': pred_test.flatten(), #raw score
                                   'y_pred': predictions_test, #round to binary
                                  })
            save_preds.to_csv(pred_f,encoding='utf-8')

        outcomes_val.to_csv(gs+'.'+tp+".metrics_val.csv", encoding='utf-8')
        outcomes_test.to_csv(gs+'.'+tp+".metrics_Ippolito.csv", encoding='utf-8')        
