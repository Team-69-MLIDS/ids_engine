from datetime import datetime
import os
import time
from uuid import uuid4
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import structlog
import xgboost as xgb
from xgboost import plot_importance

from ids_engine.src.server.data_helpers import OverallPerf, PerfMetric, Run, fig_to_base64

structlog.stdlib.recreate_defaults()
log = structlog.get_logger('lccde')

BASE_LEARNERS = [
'XGBClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'ExtraTreesClassifier'
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

    log.info("Running TreeBased...")

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
        attack_performance.update({name: perfs})

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

    df = pd.read_csv('./data/CICIDS2017_sample.csv')

    # Min-max normalization
    numeric_features = df.dtypes[df.dtypes != 'object'].index
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max()-x.min()))
    # Fill empty values by 0
    df = df.fillna(0)

    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop(['Label'],axis=1).values 
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # Train Decision Tree Classifier
    if 'DecisionTreeClassifier' in param_dict:
        dt =  DecisionTreeClassifier(**param_dict['DecisionTreeClassifier'], verbosity=-1)
    else:
        dt = DecisionTreeClassifier(verbosity=-1)

    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    record_stats('DecisionTreeClassifier', classification_report(y_test, y_pred, output_dict=True))
    
    log.debug('Accuracy of DecisionTreeClassifier: ' + str(accuracy_score(y_test, y_pred)))
    log.debug('Precision of DecisionTreeClassifier: ' + str(precision_score(y_test, y_pred, average='weighted')))
    log.debug('Recall of DecisionTreeClassifier: ' + str(recall_score(y_test, y_pred, average='weighted')))
    log.debug('Average F1 of DecisionTreeClassifier: ' + str(f1_score(y_test, y_pred, average='weighted')))
    log.debug('F1 of DecisionTreeClassifier for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
    dt_f1=f1_score(y_test, y_pred, average=None)

    # Plot the Decision Tree confusion matrix
    cm=confusion_matrix(y_test,y_pred)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor='red', fmt='.0f', ax=ax)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.title('decision tree confusion matrix')
    confusion_matrices.update({'DecisionTreeClassifier': fig_to_base64(plt)})

    # Train Random Forest Classifier
    
    if 'RandomForestClassifier' in param_dict:
        rf =  RandomForestClassifier(**param_dict['RandomForestClassifier'], verbosity=-1)
    else:
        rf = RandomForestClassifier(verbosity=-1)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    record_stats('RandomForestClassifier', classification_report(y_test, y_pred, output_dict=True))
    
    log.debug('Accuracy of RandomForestClassifier: ' + str(accuracy_score(y_test, y_pred)))
    log.debug('Precision of RandomForestClassifier: ' + str(precision_score(y_test, y_pred, average='weighted')))
    log.debug('Recall of RandomForestClassifier: ' + str(recall_score(y_test, y_pred, average='weighted')))
    log.debug('Average F1 of RandomForestClassifier: ' + str(f1_score(y_test, y_pred, average='weighted')))
    log.debug('F1 of RandomForestClassifier for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
    rf_f1=f1_score(y_test, y_pred, average=None)

    # Plot the Decision Tree confusion matrix
    cm=confusion_matrix(y_test,y_pred)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor='red', fmt='.0f', ax=ax)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.title('random forest confusion matrix')
    confusion_matrices.update({'RandomForestClassifier': fig_to_base64(plt)})

    # Train Extra Trees Classifier

    if 'ExtraTreesClassifier' in param_dict:
        xt =  ExtraTreesClassifier(**param_dict['ExtraTreesClassifier'], verbosity=-1)
    else:
        xt = ExtraTreesClassifier(verbosity=-1)

    xt.fit(X_train, y_train)
    y_pred = xt.predict(X_test)
    
    record_stats('ExtraTreesClassifier', classification_report(y_test, y_pred, output_dict=True))
    
    log.debug('Accuracy of ExtraTreesClassifier: ' + str(accuracy_score(y_test, y_pred)))
    log.debug('Precision of ExtraTreesClassifier: ' + str(precision_score(y_test, y_pred, average='weighted')))
    log.debug('Recall of ExtraTreesClassifier: ' + str(recall_score(y_test, y_pred, average='weighted')))
    log.debug('Average F1 of ExtraTreesClassifier: ' + str(f1_score(y_test, y_pred, average='weighted')))
    log.debug('F1 of ExtraTreesClassifier for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
    xt_f1=f1_score(y_test, y_pred, average=None)

    # Plot the Decision Tree confusion matrix
    cm=confusion_matrix(y_test,y_pred)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor='red', fmt='.0f', ax=ax)
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.title('extra trees confusion matrix')
    confusion_matrices.update({'ExtraTreesClassifier': fig_to_base64(plt)})

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

#!/usr/bin/env python
# coding: utf-8

# # Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles 
# This is the code for the paper entitled "[**Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles**](https://arxiv.org/pdf/1910.08635.pdf)" published in IEEE GlobeCom 2019.  
# Authors: Li Yang (liyanghart@gmail.com), Abdallah Moubayed, Ismail Hamieh, and Abdallah Shami  
# Organization: The Optimized Computing and Communications (OC2) Lab, ECE Department, Western University
# 
# If you find this repository useful in your research, please cite:  
# L. Yang, A. Moubayed, I. Hamieh and A. Shami, "Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles," 2019 IEEE Global Communications Conference (GLOBECOM), 2019, pp. 1-6, doi: 10.1109/GLOBECOM38437.2019.9013892.  

# In[1]:
import warnings
warnings.filterwarnings("ignore")
# In[2]:
# ## Read the sampled CICIDS2017 dataset
# The CICIDS2017 dataset is publicly available at: https://www.unb.ca/cic/datasets/ids-2017.html  
# Due to the large size of this dataset, the sampled subsets of CICIDS2017 is used. The subsets are in the "data" folder.  
# If you want to use this code on other datasets (e.g., CAN-intrusion dataset), just change the dataset name and follow the same steps. The models in this code are generic models that can be used in any intrusion detection/network traffic datasets.
# In[3]:
#Read dataset
df = pd.read_csv('./data/CICIDS2017.csv')
# The results in this code is based on the original CICIDS2017 dataset. Please go to cell [10] if you work on the sampled dataset. 
# ### Data sampling
# Due to the space limit of GitHub files, we sample a small-sized subset for model learning using random sampling
# In[6]:
# Randomly sample instances from majority classes
df_minor = df[(df['Label']=='WebAttack')|(df['Label']=='Bot')|(df['Label']=='Infiltration')]
df_BENIGN = df[(df['Label']=='BENIGN')]
df_BENIGN = df_BENIGN.sample(n=None, frac=0.01, replace=False, weights=None, random_state=None, axis=0)
df_DoS = df[(df['Label']=='DoS')]
df_DoS = df_DoS.sample(n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0)
df_PortScan = df[(df['Label']=='PortScan')]
df_PortScan = df_PortScan.sample(n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0)
df_BruteForce = df[(df['Label']=='BruteForce')]
df_BruteForce = df_BruteForce.sample(n=None, frac=0.2, replace=False, weights=None, random_state=None, axis=0)
# In[7]:
df_s = df_BENIGN._append(df_DoS)._append(df_PortScan)._append(df_BruteForce)._append(df_minor)
# In[8]:
df_s = df_s.sort_index()
# In[9]:
# Save the sampled dataset
df_s.to_csv('./data/CICIDS2017_sample.csv',index=0)
# ### Preprocessing (normalization and padding values)
# In[10]:
df = pd.read_csv('./data/CICIDS2017_sample.csv')
# In[11]:
# Min-max normalization
numeric_features = df.dtypes[df.dtypes != 'object'].index
df[numeric_features] = df[numeric_features].apply(
    lambda x: (x - x.min()) / (x.max()-x.min()))
# Fill empty values by 0
df = df.fillna(0)
# ### split train set and test set
# In[18]:
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
X = df.drop(['Label'],axis=1).values 
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
# In[19]:
X_train.shape
# In[20]:
pd.Series(y_train).value_counts()
# ### Oversampling by SMOTE
# In[21]:
from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500}) # Create 1500 samples for the minority class "4"


# In[22]:


X_train, y_train = smote.fit_resample(X_train, y_train.astype('int'))


# In[23]:


pd.Series(y_train).value_counts()


# ## Machine learning model training

# ### Training four base learners: decision tree, random forest, extra trees, XGBoost

# In[11]:


# Decision tree training and prediction
dt = DecisionTreeClassifier(random_state = 0)
dt.fit(X_train,y_train) 
dt_score=dt.score(X_test,y_test)
y_predict=dt.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# In[12]:


dt_train=dt.predict(X_train)
dt_test=dt.predict(X_test)


# In[13]:
# Random Forest training and prediction
rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train,y_train) 
rf_score=rf.score(X_test,y_test)
y_predict=rf.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
# In[14]:
rf_train=rf.predict(X_train)
rf_test=rf.predict(X_test)
# In[15]:
# Extra trees training and prediction
et = ExtraTreesClassifier(random_state = 0)
et.fit(X_train,y_train) 
et_score=et.score(X_test,y_test)
y_predict=et.predict(X_test)
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
# In[16]:
et_train=et.predict(X_train)
et_test=et.predict(X_test)
# In[17]:
# XGboost training and prediction
xg = xgb.XGBClassifier(n_estimators = 10)
xg.fit(X_train,y_train)
xg_score=xg.score(X_test,y_test)
y_predict=xg.predict(X_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
# In[18]:
xg_train=xg.predict(X_train)
xg_test=xg.predict(X_test)
# ### Stacking model construction (ensemble for 4 base learners)

# In[19]:


# Use the outputs of 4 base models to construct a new ensemble model
base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
     'ExtraTrees': et_train.ravel(),
     'XgBoost': xg_train.ravel(),
    })
base_predictions_train.head(5)


# In[20]:


dt_train=dt_train.reshape(-1, 1)
et_train=et_train.reshape(-1, 1)
rf_train=rf_train.reshape(-1, 1)
xg_train=xg_train.reshape(-1, 1)
dt_test=dt_test.reshape(-1, 1)
et_test=et_test.reshape(-1, 1)
rf_test=rf_test.reshape(-1, 1)
xg_test=xg_test.reshape(-1, 1)


# In[21]:


x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)


# In[22]:


stk = xgb.XGBClassifier().fit(x_train, y_train)


# In[23]:
y_predict=stk.predict(x_test)
y_true=y_test
stk_score=accuracy_score(y_true,y_predict)
print('Accuracy of Stacking: '+ str(stk_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of Stacking: '+(str(precision)))
print('Recall of Stacking: '+(str(recall)))
print('F1-score of Stacking: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
# ## Feature Selection
# ### Feature importance
# In[24]:
# Save the feature importance lists generated by four tree-based algorithms
dt_feature = dt.feature_importances_
rf_feature = rf.feature_importances_
et_feature = et.feature_importances_
xgb_feature = xg.feature_importances_
# In[25]:
# calculate the average importance value of each feature
avg_feature = (dt_feature + rf_feature + et_feature + xgb_feature)/4
# In[26]:
feature=(df.drop(['Label'],axis=1)).columns.values
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True))
# In[27]:
f_list = sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True)
# In[29]:
# Select the important features from top-importance to bottom-importance until the accumulated importance reaches 0.9 (out of 1)
Sum = 0
fs = []
for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])
    if Sum>=0.9:
        break        


# In[30]:


X_fs = df[fs].values


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X_fs,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[32]:


X_train.shape


# In[33]:


pd.Series(y_train).value_counts()


# ### Oversampling by SMOTE

# In[34]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500})


# In[35]:
X_train, y_train = smote.fit_resample(X_train, y_train)
# In[36]:


pd.Series(y_train).value_counts()


# ## Machine learning model training after feature selection

# In[37]:


dt = DecisionTreeClassifier(random_state = 0)
dt.fit(X_train,y_train) 
dt_score=dt.score(X_test,y_test)
y_predict=dt.predict(X_test)
y_true=y_test
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# In[38]:


dt_train=dt.predict(X_train)
dt_test=dt.predict(X_test)


# In[39]:


rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train,y_train) # modelin veri üzerinde öğrenmesi fit fonksiyonuyla yapılıyor
rf_score=rf.score(X_test,y_test)
y_predict=rf.predict(X_test)
y_true=y_test
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# In[40]:


rf_train=rf.predict(X_train)
rf_test=rf.predict(X_test)


# In[41]:


et = ExtraTreesClassifier(random_state = 0)
et.fit(X_train,y_train) 
et_score=et.score(X_test,y_test)
y_predict=et.predict(X_test)
y_true=y_test
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# In[42]:


et_train=et.predict(X_train)
et_test=et.predict(X_test)


# In[43]:


xg = xgb.XGBClassifier(n_estimators = 10)
xg.fit(X_train,y_train)
xg_score=xg.score(X_test,y_test)
y_predict=xg.predict(X_test)
y_true=y_test
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# In[44]:


xg_train=xg.predict(X_train)
xg_test=xg.predict(X_test)


# ### Stacking model construction

# In[45]:


base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
     'ExtraTrees': et_train.ravel(),
     'XgBoost': xg_train.ravel(),
    })
base_predictions_train.head(5)


# In[46]:


dt_train=dt_train.reshape(-1, 1)
et_train=et_train.reshape(-1, 1)
rf_train=rf_train.reshape(-1, 1)
xg_train=xg_train.reshape(-1, 1)
dt_test=dt_test.reshape(-1, 1)
et_test=et_test.reshape(-1, 1)
rf_test=rf_test.reshape(-1, 1)
xg_test=xg_test.reshape(-1, 1)


# In[47]:


x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)


# In[48]:


stk = xgb.XGBClassifier().fit(x_train, y_train)
y_predict=stk.predict(x_test)
y_true=y_test
stk_score=accuracy_score(y_true,y_predict)
print('Accuracy of Stacking: '+ str(stk_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
print('Precision of Stacking: '+(str(precision)))
print('Recall of Stacking: '+(str(recall)))
print('F1-score of Stacking: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# In[ ]:




