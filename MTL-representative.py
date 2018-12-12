# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:07:42 2018

@author: hwang
"""

import numpy as np
import pandas as pd

import tensorflow
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn import metrics
from sklearn.model_selection import KFold
import os

from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)



def intersect(a, b):
    return list(set(a) & set(b))


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
    

dataset = ['TMG',]


tp = ['Biliary.Hyperplasia','Fibrosis','Necrosis']

path = 'Data' #datasets should be here

for gs in dataset:    
    print("data set {}".format(gs))
    tr = pd.read_csv(os.path.join(path, gs+'.tr.csv'),index_col=0)
    test1 = pd.read_csv(os.path.join(path,gs+'.tr_del.csv'),index_col=0)         
    test2 = pd.read_csv(os.path.join(path,gs+'.Ippolito.csv'),index_col=0)  
    X_tr = tr.drop(tp, axis = 1)
    sampleID_tr = tr.index.values
    X_tr = X_tr.values
    y_tr = tr[tp]
    y_tr = y_tr.values

    X_test2 = test2.drop(tp, axis = 1)
    sampleID_test2 = test2.index.values
    X_test2 = X_test2.values
    y_test2 = test2[tp]
    y_test2 = y_test2.values
    
    outcomes_test2 = pd.DataFrame()
    outcomes_train = pd.DataFrame()
    outcomes_val = pd.DataFrame()
    
    sub_nn_name = tp[:]
    sub_nn_name.insert(0,'shared')
    ly_sub = [2,1,1,1]
    nd_sub = [250,100,100,100]
    n_node = dict(zip(sub_nn_name, nd_sub))

    n_batch = 40
    n_epoch = 1000
    n_dropout = 0.8
    rd = 0
    n_run=10
    kf = KFold(n_splits=n_run, random_state=255, shuffle=True)
    
    for train_index, val_index in kf.split(X_tr, y_tr):
        rd=rd+1
        X_train, X_val = X_tr[train_index], X_tr[val_index]
        y_train, y_val = y_tr[train_index], y_tr[val_index]
        
        sampleID_train = list(sampleID_tr[list(train_index)]) 
        sampleID_val = list(sampleID_tr[list(val_index)])
        
        inputs = Input(shape=(X_train.shape[1],))
        #the first shared layer
        shared = Dense(n_node['shared'],
                       kernel_initializer='normal', 
                       activation='relu')(inputs)
        shared = Dropout(n_dropout)(shared)

        #the second shared layer
        shared = Dense(n_node['shared'],
                       kernel_initializer='normal', 
                       activation='relu')(shared)
        shared = Dropout(n_dropout)(shared)
        
        #one specfic layer for each endpoint
        sub1 = Dense(n_node['Biliary.Hyperplasia'],
                     kernel_initializer='normal', 
                     activation='relu')(shared)
        sub1 = Dropout(n_dropout)(sub1)

        sub2 = Dense(n_node['Fibrosis'],
                     kernel_initializer='normal', 
                     activation='relu')(shared)
        sub2 = Dropout(n_dropout)(sub2)
        sub3 = Dense(n_node['Necrosis'],
                     kernel_initializer='normal', 
                     activation='relu')(shared)
        sub3 = Dropout(n_dropout)(sub3)

        #outlayers
        out1 = Dense(1,kernel_initializer='normal', activation='sigmoid',name='Biliary.Hyperplasia')(sub1)
        out2 = Dense(1,kernel_initializer='normal', activation='sigmoid',name='Fibrosis')(sub2)
        out3 = Dense(1,kernel_initializer='normal', activation='sigmoid',name='Necrosis')(sub3)
        
        model = Model(inputs=inputs, outputs=[out1,out2,out3])

        mod_f = gs+".model-best."+str(rd)+".json"
        model_json = model.to_json()
        with open(mod_f, "w") as json_file:
            json_file.write(model_json)
        
        model.compile(optimizer='adam',
                      loss={'Biliary.Hyperplasia':'binary_crossentropy',
                            'Fibrosis':'binary_crossentropy',
                            'Necrosis':'binary_crossentropy',
                            },
                      loss_weights={'Biliary.Hyperplasia':1,
                            'Fibrosis':1,
                            'Necrosis':1,
                            },
                      )
        
        early_stopping = EarlyStopping(
                                       monitor='val_loss', 
                                       patience=100, verbose=0, mode='auto')
        filepath=gs+".weights-best."+str(rd)+".hdf5"

        checkpoint = ModelCheckpoint(filepath, 
                                     monitor='val_loss',
                                     verbose=0, 
                                     save_best_only=True,
                                     mode='auto')
            
        callbacks_list = [
                          early_stopping,
                          checkpoint,
                          ]
        
        weight = {}
        from sklearn.utils import class_weight
        for i in range(len(tp)):
            label = y_train[:,i]
            label_list = label.tolist()
            sub_w = class_weight.compute_class_weight('balanced',
                                                     np.unique(label_list),
                                                     label_list)
            weight[tp[i]] = dict(enumerate(sub_w))
        
        train_history = model.fit(X_train, 
                  {'Biliary.Hyperplasia':y_train[:,0],
                   'Fibrosis':y_train[:,1],
                   'Necrosis':y_train[:,2]
                  },
                  epochs=n_epoch, batch_size=n_batch,  verbose=0,
                  class_weight=weight,
                  validation_split=0.1,
                  callbacks = callbacks_list
                 )

        json_file = open(mod_f, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        best_model = model_from_json(loaded_model_json)
        best_model.load_weights(filepath)
        best_model.compile(optimizer='adam',
                      loss={'Biliary.Hyperplasia':'binary_crossentropy',
                            'Fibrosis':'binary_crossentropy',
                            'Necrosis':'binary_crossentropy',
                            },
                      loss_weights={'Biliary.Hyperplasia':1,
                            'Fibrosis':1,
                            'Necrosis':1,
                            },
                      )
        
        pred_train = best_model.predict(X_train)
        pred_round_train = []
        for i in range(len(pred_train)):
            prnd = [round(x[0]) for x in pred_train[i]]
            pred_round_train.append(prnd)
        predictions_train = np.array(pred_round_train).transpose()
        
        print("train")
        report_train = metrics.classification_report(y_train, predictions_train,
                                                     target_names=tp)
        
        print(report_train)
        
        mcc_train = {}
        mcc_train['metric'] = 'MCC'
        f1_train = {}
        f1_train['metric'] = 'F1'
        
        for i in range(y_train.shape[1]):
            mcc_train[tp[i]] = metrics.matthews_corrcoef(y_train[:,i], predictions_train[:,i])
            f1_train[tp[i]] = metrics.f1_score(y_train[:,i], predictions_train[:,i])
        
        outcomes_train = outcomes_train.append({
                    'metric':mcc_train['metric'],
                    'Biliary.Hyperplasia':mcc_train['Biliary.Hyperplasia'],
                    'Fibrosis':mcc_train['Fibrosis'],
                    'Necrosis':mcc_train['Necrosis'],
                    }
                ,ignore_index=True)
        
        outcomes_train = outcomes_train.append({
                    'metric':f1_train['metric'],
                    'Biliary.Hyperplasia':f1_train['Biliary.Hyperplasia'],
                    'Fibrosis':f1_train['Fibrosis'],
                    'Necrosis':f1_train['Necrosis'],
                    }
                ,ignore_index=True)
    
        
        pred_val = best_model.predict(X_val)
        pred_round_val = []
        for i in range(len(pred_val)):
            prnd = [round(x[0]) for x in pred_val[i]]
            pred_round_val.append(prnd)
        predictions_val = np.array(pred_round_val).transpose()
        
        val_pred_array = np.concatenate([y_val,predictions_val],axis=1)
        val_pred_df = pd.DataFrame(data=val_pred_array,
                                   index=sampleID_val,
                                   columns=['Biliary.Hyperplasia_exp','Fibrosis_exp','Necrosis_exp',
                                                                'Biliary.Hyperplasia_pred','Fibrosis_pred','Necrosis_pred',])
        val_pred_df.to_csv(gs+"."+str(rd)+".val.predictions.csv")

        print("val")
        report_val = metrics.classification_report(y_val, predictions_val,
                                                     target_names=tp)
        
        print(report_val)
            
        mcc_val = {}
        mcc_val['metric'] = 'MCC'
        f1_val = {}
        f1_val['metric'] = 'F1'
        
        for i in range(y_val.shape[1]):
            mcc_val[tp[i]] = metrics.matthews_corrcoef(y_val[:,i], predictions_val[:,i])
            f1_val[tp[i]] = metrics.f1_score(y_val[:,i], predictions_val[:,i])
        
        outcomes_val = outcomes_val.append({
                    'metric':mcc_val['metric'],
                    'Biliary.Hyperplasia':mcc_val['Biliary.Hyperplasia'],
                    'Fibrosis':mcc_val['Fibrosis'],
                    'Necrosis':mcc_val['Necrosis'],
                    }
                ,ignore_index=True)
        
        outcomes_val = outcomes_val.append({
                    'metric':f1_val['metric'],
                    'Biliary.Hyperplasia':f1_val['Biliary.Hyperplasia'],
                    'Fibrosis':f1_val['Fibrosis'],
                    'Necrosis':f1_val['Necrosis'],
                    }
                ,ignore_index=True)
        
        pred_test2 = best_model.predict(X_test2)
        pred_round_test2 = []
        for i in range(len(pred_test2)):
            prnd = [round(x[0]) for x in pred_test2[i]]
            pred_round_test2.append(prnd)
        predictions_test2 = np.array(pred_round_test2).transpose()

        test2_pred_array = np.concatenate([y_test2,predictions_test2],axis=1)
        test2_pred_df = pd.DataFrame(data=test2_pred_array,
                                   index=sampleID_test2,
                                   columns=['Biliary.Hyperplasia_exp','Fibrosis_exp','Necrosis_exp',
                                                                'Biliary.Hyperplasia_pred','Fibrosis_pred','Necrosis_pred',])
        test2_pred_df.to_csv(gs+"."+str(rd)+".Ippolito.predictions.csv")
        
        report_test2 = metrics.classification_report(y_test2, predictions_test2,
                                                     target_names=tp)
        print("test2")
        print(report_test2)
        
        mcc = {}
        mcc['metric'] = 'MCC'
        f1 = {}
        f1['metric'] = 'F1'
        
        for i in range(y_test2.shape[1]):
            mcc[tp[i]] = metrics.matthews_corrcoef(y_test2[:,i], predictions_test2[:,i])
            f1[tp[i]] = metrics.f1_score(y_test2[:,i], predictions_test2[:,i])
        
        outcomes_test2 = outcomes_test2.append({
                    'metric':mcc['metric'],
                    'Biliary.Hyperplasia':mcc['Biliary.Hyperplasia'],
                    'Fibrosis':mcc['Fibrosis'],
                    'Necrosis':mcc['Necrosis'],
                    }
                ,ignore_index=True)
        outcomes_test2 = outcomes_test2.append({
                    'metric':f1['metric'],
                    'Biliary.Hyperplasia':f1['Biliary.Hyperplasia'],
                    'Fibrosis':f1['Fibrosis'],
                    'Necrosis':f1['Necrosis'],
                    }
                ,ignore_index=True)

    outcomes_train.to_csv(gs+".outcomes.train.csv") 
    outcomes_test2.to_csv(gs+".outcomes.Ippolito.csv")
    outcomes_val.to_csv(gs+".outcomes.val.csv")
