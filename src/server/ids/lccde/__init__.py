# This is based on `LCCDE_IDS_GlobeCom22.ipynb`
from datetime import datetime
from uuid import uuid4
import warnings
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb
import time
from river import stream
from statistics import mode
from imblearn.over_sampling import SMOTE
import structlog
import base64
import json
import logging
from server.data_helpers import Run, PerfMetric, OverallPerf, fig_to_base64
import os 

structlog.stdlib.recreate_defaults()
log = structlog.get_logger('lccde')




BASE_LEARNERS = [
        'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier'
]

def train_model(run_tag: str, 
                param_dict: dict, 
                dataset: None|str) -> Run:

    matplotlib.set_loglevel('critical')
    warnings.filterwarnings("ignore", category=UserWarning)

    detection_model_name = 'lccde'
    timestamp = datetime.now()
    confusion_matrices : dict[str, str] = dict()
    attack_performance : dict[str, dict[str, PerfMetric]] = dict()
    overall_performance : dict[str, OverallPerf] = dict()

    log.info('Running LCCD...')
    # load the data
    if dataset is not None: 
        ds_path= os.path.join(os.path.curdir, 'data', dataset)
        if not os.path.exists(ds_path):
            log.error('path does not exit: ', ds_path)
            time.sleep(50000)
        df = pd.read_csv(ds_path)
    else:
        df = pd.read_csv('./data/CICIDS2017_sample_km.csv')
        dataset = './data/CICIDS2017_sample_km.csv'

    log.debug(df)

    log.debug(df.Label.value_counts())

    # split train set and test set
    X = df.drop(['Label'],axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0) 	#shuffle=False
    log.debug(X_train, X_test, y_train, y_test)

    # SMOTE to solve class imbalance
    pd.Series(y_train).value_counts()
    smote=SMOTE(n_jobs=-1, sampling_strategy={2:1000,4:1000})
    X_train, y_train = smote.fit_resample(X_train, y_train)


    ## Training three base learners lightgbm, xgboost, catboost

    # Train the LightGBM algorithm
    if 'LGBMClassifier' in param_dict:
        lg = lgb.LGBMClassifier(**param_dict['LGBMClassifier'], verbosity=-1)
    else: 
        lg = lgb.LGBMClassifier(verbosity=-1)

    lg.fit(X_train, y_train)
    y_pred = lg.predict(X_test)

    def record_stats(name, report): 
        print(report)
        overall_performance.update({
            name: OverallPerf(
                accuracy= report['accuracy'],
                macro_avg_precision = report['macro avg']['precision'],
                macro_avg_recall = report['macro avg']['recall'],
                macro_avg_f1_score = report['macro avg']['f1-score'],
                macro_avg_support = report['macro avg']['support'],
                weighted_avg_precision = report['weighted avg']['precision'],
                weighted_avg_recall = report['weighted avg']['recall'],
                weighted_avg_f1_score = report['weighted avg']['f1-score'],
                weighted_avg_support = report['weighted avg']['support'],
            )
        })
        perfs: dict[str, PerfMetric] = dict()
        dont_read = [
            'accuracy','macro avg','weighted avg'
        ]
        for class_name, stats in report.items(): 
            if class_name not in dont_read:
                print(class_name, stats)
                perfs.update({class_name: PerfMetric(
                    support=stats['support'],
                    f1_score=stats['f1-score'],
                    recall =stats['recall'],
                    precision =stats['precision'],
                )})

        # for i in range(6):
        #     atk = report[f'{i}']
        #     perfs.append(PerfMetric(
        #         support=atk['support'],
        #         f1_score=atk['f1-score'],
        #         recall =atk['recall'],
        #         precision =atk['precision'],
        #     ))
        attack_performance.update({name: perfs})

    record_stats('LGBMClassifier', classification_report(y_test, y_pred, output_dict=True))



    log.debug('Accuracy of LightGBM: ' + str(accuracy_score(y_test, y_pred)))
    log.debug('Precision of LightGBM: ' + str(precision_score(y_test, y_pred, average='weighted')))
    log.debug('Recall of LightGBM: ' + str(recall_score(y_test, y_pred, average='weighted')))
    log.debug('Average F1 of LightGBM: ' + str(f1_score(y_test, y_pred, average='weighted')))
    log.debug('F1 of LightGBM for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
    lg_f1=f1_score(y_test, y_pred, average=None)

    # Plot the confusion matrix
    cm=confusion_matrix(y_test,y_pred)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor='red', fmt='.0f', ax=ax)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.title('lightgbm confusion matrix')
    confusion_matrices.update({'LGBMClassifier': fig_to_base64(plt)})

    # log.debug(fig_to_base64(plt))

    # plt.savefig('./lightgbm_confusion_matrix.png')


    	# Train the XGBoost algorithm
    if 'XGBClassifier' in param_dict:
        xg = xgb.XGBClassifier(**param_dict['XGBClassifier'])
    else:
        xg = xgb.XGBClassifier()


    X_train_x = X_train.values
    X_test_x = X_test.values

    xg.fit(X_train_x, y_train)

    y_pred = xg.predict(X_test_x)
    log.debug(classification_report(y_test,y_pred))
    log.debug('Accuracy of XGBoost: '+ str(accuracy_score(y_test, y_pred)))
    log.debug('Precision of XGBoost: '+ str(precision_score(y_test, y_pred, average='weighted')))
    log.debug('Recall of XGBoost: '+ str(recall_score(y_test, y_pred, average='weighted')))
    log.debug('Average F1 of XGBoost: '+ str(f1_score(y_test, y_pred, average='weighted')))
    log.debug('F1 of XGBoost for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
    xg_f1=f1_score(y_test, y_pred, average=None)

    print(json.dumps(classification_report(y_test, y_pred, output_dict=True), indent=4))
    record_stats('XGBClassifier', classification_report(y_test, y_pred, output_dict=True))
    	# Plot the confusion matrix
    cm=confusion_matrix(y_test,y_pred)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor='red',fmt='.0f',ax=ax)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.title('xgboost confusion matrix')
    confusion_matrices.update({'XGBClassifier': fig_to_base64(plt)})
    # plt.savefig('./xgboost_confusion_matrix.png')


	# Train the CatBoost algorithm
    if 'CatBoostClassifier' in param_dict:
        cb = cbt.CatBoostClassifier(verbose=0,boosting_type='Plain', **param_dict['CatBoostClassifier'])
    else:
        cb = cbt.CatBoostClassifier(verbose=0,boosting_type='Plain')


    cb.fit(X_train, y_train)
    y_pred = cb.predict(X_test)
    log.debug(classification_report(y_test,y_pred))
    log.debug('Accuracy of CatBoost: '+ str(accuracy_score(y_test, y_pred)))
    log.debug('Precision of CatBoost: '+ str(precision_score(y_test, y_pred, average='weighted')))
    log.debug('Recall of CatBoost: '+ str(recall_score(y_test, y_pred, average='weighted')))
    log.debug('Average F1 of CatBoost: '+ str(f1_score(y_test, y_pred, average='weighted')))
    log.debug('F1 of CatBoost for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
    cb_f1=f1_score(y_test, y_pred, average=None)
    print('\n\n\n\n\n CATBOOST YT AND YP', y_test,'='*20, y_pred)

    record_stats('CatBoostClassifier', classification_report(y_test, y_pred, output_dict=True))

	# Plot the confusion matrix
    cm=confusion_matrix(y_test,y_pred)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor='red',fmt='.0f',ax=ax)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.title('catboost confusion matrix')
    confusion_matrices.update({'CatBoostClassifier': fig_to_base64(plt)})
    # plt.savefig('./catboost_confusion_matrix.png')

	#	# Proposed ensemble model: Leader Class and Confidence Decision Ensemble (LCCDE)

	#	#	# Find the best-performing (leading) model for each type of attack among the three ML models
    model=[]
    for i in range(len(lg_f1)):
        if max(lg_f1[i],xg_f1[i],cb_f1[i]) == lg_f1[i]:
            model.append(lg)
        elif max(lg_f1[i],xg_f1[i],cb_f1[i]) == xg_f1[i]:
            model.append(xg)
        else:
            model.append(cb)

    log.debug(model)


    def LCCDE(X_test, y_test, m1, m2, m3):
        i = 0
        t = []
        m = []
        yt = []
        yp = []
        l = []
        pred_l = []
        pro_l = []

        	# For each class (normal or a type of attack), find the leader model
        for xi, yi in stream.iter_pandas(X_test, y_test):

            xi2=np.array(list(xi.values()))
            y_pred1 = m1.predict(xi2.reshape(1, -1))      	# model 1 (LightGBM) makes a prediction on text sample xi
            y_pred1 = int(y_pred1[0])
            y_pred2 = m2.predict(xi2.reshape(1, -1))      	# model 2 (XGBoost) makes a prediction on text sample xi
            y_pred2 = int(y_pred2[0])
            y_pred3 = m3.predict(xi2.reshape(1, -1))      	# model 3 (Catboost) makes a prediction on text sample xi
            y_pred3 = int(y_pred3[0])

            p1 = m1.predict_proba(xi2.reshape(1, -1))     	# The prediction probability (confidence) list of model 1 
            p2 = m2.predict_proba(xi2.reshape(1, -1))     	# The prediction probability (confidence) list of model 2  
            p3 = m3.predict_proba(xi2.reshape(1, -1))     	# The prediction probability (confidence) list of model 3  

            	# Find the highest prediction probability among all classes for each ML model
            y_pred_p1 = np.max(p1)
            y_pred_p2 = np.max(p2)
            y_pred_p3 = np.max(p3)

            if y_pred1 == y_pred2 == y_pred3: 	# If the predicted classes of all the three models are the same
                y_pred = y_pred1 	# Use this predicted class as the final predicted class

            elif y_pred1 != y_pred2 != y_pred3: 	# If the predicted classes of all the three models are different
                	# For each prediction model, check if the predicted class’s original ML model is the same as its leader model
                if model[y_pred1]==m1: 	# If they are the same and the leading model is model 1 (LightGBM)
                    l.append(m1)
                    pred_l.append(y_pred1) 	# Save the predicted class
                    pro_l.append(y_pred_p1) 	# Save the confidence

                if model[y_pred2]==m2: 	# If they are the same and the leading model is model 2 (XGBoost)
                    l.append(m2)
                    pred_l.append(y_pred2)
                    pro_l.append(y_pred_p2)

                if model[y_pred3]==m3: 	# If they are the same and the leading model is model 3 (CatBoost)
                    l.append(m3)
                    pred_l.append(y_pred3)
                    pro_l.append(y_pred_p3)

                if len(l)==0: 	# Avoid empty probability list
                    pro_l=[y_pred_p1,y_pred_p2,y_pred_p3]

                elif len(l)==1: 	# If only one pair of the original model and the leader model for each predicted class is the same
                    y_pred=pred_l[0] 	# Use the predicted class of the leader model as the final prediction class

                else: 	# If no pair or multiple pairs of the original prediction model and the leader model for each predicted class are the same
                    max_p = max(pro_l) 	# Find the highest confidence
                    
                    	# Use the predicted class with the highest confidence as the final prediction class
                    if max_p == y_pred_p1:
                        y_pred = y_pred1
                    elif max_p == y_pred_p2:
                        y_pred = y_pred2
                    else:
                        y_pred = y_pred3  
            
            else: 	# If two predicted classes are the same and the other one is different
                n = mode([y_pred1,y_pred2,y_pred3]) 	# Find the predicted class with the majority vote
                y_pred = model[n].predict(xi2.reshape(1, -1)) 	# Use the predicted class of the leader model as the final prediction class
                y_pred = y_pred[0]

            yt.append(yi)
            yp.append(y_pred) 	# Save the predicted classes for all tested samples
        return yt, yp

	# Implementing LCCDE
    yt, yp = LCCDE(X_test, y_test, m1 = lg, m2 = xg, m3 = cb)

    print('YT AND YP', yt, yp)
	# The performance of the proposed lCCDE model,
    log.debug('Accuracy of LCCDE: '+ str(accuracy_score(yt, yp)))
    log.debug('Precision of LCCDE: '+ str(precision_score(yt, yp, average='weighted')))
    log.debug('Recall of LCCDE: '+ str(recall_score(yt, yp, average='weighted')))
    log.debug('Average F1 of LCCDE: '+ str(f1_score(yt, yp, average='weighted')))
    log.debug('F1 of LCCDE for each type of attack: '+ str(f1_score(yt, yp, average=None)))
    record_stats('lccde', classification_report(yt, yp, output_dict=True))
    cm=confusion_matrix(yt, yp)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor='red', fmt='.0f', ax=ax)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.title('lccde confusion matrix')
    confusion_matrices.update({'lccde': fig_to_base64(plt)})

    run = Run(
        id=str(uuid4()),
        run_tag=run_tag,
        detection_model_name=detection_model_name,
        learner_configuration=param_dict,
        learner_performance_per_attack=attack_performance,
        timestamp=str(timestamp),
        confusion_matrices=confusion_matrices,
        dataset=dataset,
        learner_overalls=overall_performance
        )
    return run
