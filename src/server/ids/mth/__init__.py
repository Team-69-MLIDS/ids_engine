# coding: utf-8

# # MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles
# This is the code for the paper entitled "[**MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles**](https://arxiv.org/pdf/2105.13289.pdf)" accepted in IEEE Internet of Things Journal.  
# Authors: Li Yang (liyanghart@gmail.com), Abdallah Moubayed, and Abdallah Shami  
# Organization: The Optimized Computing and Communications (OC2) Lab, ECE Department, Western University
# 
# If you find this repository useful in your research, please cite:  
# L. Yang, A. Moubayed, and A. Shami, “MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles,” IEEE Internet of Things Journal, vol. 9, no. 1, pp. 616-632, Jan.1, 2022.

# ## Import libraries

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
import matplotlib
from datetime import datetime
from server.data_helpers import Run, PerfMetric, OverallPerf, fig_to_base64
import time
from server.ids.mth.FCBF_module import FCBF, FCBFK, FCBFiP, get_i

BASE_LEARNERS = [
    'XGBClassifier', 
    'RandomForestClassifier',
    'DecisionTreeClassifier',
    'ExtraTreesClassifier'
]


def train_model(run_tag: str, 
                param_dict: dict, 
                dataset: None|str) -> Run:

    matplotlib.set_loglevel('critical')
    dataset= './data/CICIDS2017_sample_km.csv'
    matplotlib.set_loglevel('critical')
    warnings.filterwarnings("ignore", category=UserWarning)

    detection_model_name = 'mth'
    timestamp = datetime.now()
    confusion_matrices : dict[str, str] = dict()
    attack_performance : dict[str, dict[str, PerfMetric]] = dict()
    overall_performance : dict[str, OverallPerf] = dict()

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

# ## Read the sampled CICIDS2017 dataset
# The CICIDS2017 dataset is publicly available at: https://www.unb.ca/cic/datasets/ids-2017.html  
# Due to the large size of this dataset, the sampled subsets of CICIDS2017 is used. The subsets are in the "data" folder.  
# If you want to use this code on other datasets (e.g., CAN-intrusion dataset), just change the dataset name and follow the same steps. The models in this code are generic models that can be used in any intrusion detection/network traffic datasets.

# In[3]:


#Read dataset
    
    df = pd.read_csv('./data/CICIDS2017.csv') 
    # df = pd.read_csv(dataset or '')  # here, we want to provide either the CICIDS or CAN-intrusion dataset
# The results in this code is based on the original CICIDS2017 dataset. Please go to cell [21] if you work on the sampled dataset. 


# Z-score normalization
    features = df.dtypes[df.dtypes != 'object'].index
    df[features] = df[features].apply(
        lambda x: (x - x.mean()) / (x.std()))
# Fill empty values by 0
    df = df.fillna(0)


# ### Data sampling
# Due to the space limit of GitHub files and the large size of network traffic data, we sample a small-sized subset for model learning using **k-means cluster sampling**

# In[7]:


    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])


# In[8]:


    df.Label.value_counts()


# In[9]:


# retain the minority class instances and sample the majority class instances
    df_minor = df[(df['Label']==6)|(df['Label']==1)|(df['Label']==4)]
    df_major = df.drop(df_minor.index)


# In[10]:


    X = df_major.drop(['Label'],axis=1) 
    y = df_major.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)


# In[11]:


# use k-means to cluster the data samples and select a proportion of data from each cluster
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)


# In[12]:


    klabel=kmeans.labels_
    df_major['klabel']=klabel


# In[13]:


    df_major['klabel'].value_counts()


# In[14]:


    cols = list(df_major)
    cols.insert(78, cols.pop(cols.index('Label')))
    df_major = df_major.loc[:, cols]


# In[15]:




# In[16]:


    def typicalSampling(group):
        name = group.name
        frac = 0.008
        return group.sample(frac=frac)

    result = df_major.groupby(
        'klabel', group_keys=False
    ).apply(typicalSampling)


# In[17]:


    result['Label'].value_counts()


# In[18]:


    result


# In[19]:


    result = result.drop(['klabel'],axis=1)
    result = result._append(df_minor)


# In[20]:


    result.to_csv('./data/CICIDS2017_sample_km.csv',index=0)


# ### split train set and test set

# In[21]:


# Read the sampled dataset
    df=pd.read_csv('./data/CICIDS2017_sample_km.csv')


# In[22]:


    X = df.drop(['Label'],axis=1).values
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)


# In[23]:


    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)


# ## Feature engineering

# ### Feature selection by information gain

# In[24]:


    from sklearn.feature_selection import mutual_info_classif
    importances = mutual_info_classif(X_train, y_train)


# In[25]:


# calculate the sum of importance scores
    f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])


# In[26]:


# select the important features from top to bottom until the accumulated importance reaches 90%
    f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
    Sum2 = 0
    fs = []
    for i in range(0, len(f_list2)):
        Sum2 = Sum2 + f_list2[i][0]
        fs.append(f_list2[i][1])
        if Sum2>=0.9:
            break        


# In[27]:


    X_fs = df[fs].values


# In[28]:


    X_fs.shape


# ### Feature selection by Fast Correlation Based Filter (FCBF)
# 
# The module is imported from the GitHub repo: https://github.com/SantiagoEG/FCBF_module

# In[29]:


    from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
    fcbf = FCBFK(k = 20)

#fcbf.fit(X_fs, y)


# In[30]:


    X_fss = fcbf.fit_transform(X_fs,y)


# ### Re-split train & test sets after feature selection
    X_train, X_test, y_train, y_test = train_test_split(X_fss,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)

# ### SMOTE to solve class-imbalance

# In[35]:


    from imblearn.over_sampling import SMOTE
    smote=SMOTE(n_jobs=-1,sampling_strategy={2:1000,4:1000})


# In[36]:


    X_train, y_train = smote.fit_resample(X_train, y_train)


# In[37]:


    pd.Series(y_train).value_counts()


# ## Machine learning model training

# ### Training four base learners: decision tree, random forest, extra trees, XGBoost

# #### Apply XGBoost

# In[58]:


    xg = xgb.XGBClassifier(**param_dict['XGBClassifier']) if 'XGBClassifier' in param_dict else xgb.XGBClassifier(n_estimators = 10)
        
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
    record_stats('XGBClassifier', classification_report(y_true, y_predict, output_dict=True))
    time.sleep(20000)
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    # 




# In[114]:


    xg = xgb.XGBClassifier(learning_rate= 0.7340229699980686, n_estimators = 70, max_depth = 14)
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
    # 


# In[115]:


    xg_train=xg.predict(X_train)
    xg_test=xg.predict(X_test)


# #### Apply RF

# In[103]:


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
    # 



# In[104]:


    rf_hpo = RandomForestClassifier(n_estimators = 71, min_samples_leaf = 1, max_depth = 46, min_samples_split = 9, max_features = 20, criterion = 'entropy')
    rf_hpo.fit(X_train,y_train)
    rf_score=rf_hpo.score(X_test,y_test)
    y_predict=rf_hpo.predict(X_test)
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
    # 


# In[105]:


    rf_train=rf_hpo.predict(X_train)
    rf_test=rf_hpo.predict(X_test)


# #### Apply DT

# In[100]:


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
    # 


# #### Hyperparameter optimization (HPO) of decision tree using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
# Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms



# In[101]:


    dt_hpo = DecisionTreeClassifier(min_samples_leaf = 2, max_depth = 47, min_samples_split = 3, max_features = 19, criterion = 'gini')
    dt_hpo.fit(X_train,y_train)
    dt_score=dt_hpo.score(X_test,y_test)
    y_predict=dt_hpo.predict(X_test)
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
    # 


# In[102]:


    dt_train=dt_hpo.predict(X_train)
    dt_test=dt_hpo.predict(X_test)


# #### Apply ET

# In[106]:


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
    # 


# In[108]:


    et_hpo = ExtraTreesClassifier(n_estimators = 53, min_samples_leaf = 1, max_depth = 31, min_samples_split = 5, max_features = 20, criterion = 'entropy')
    et_hpo.fit(X_train,y_train) 
    et_score=et_hpo.score(X_test,y_test)
    y_predict=et_hpo.predict(X_test)
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
    # 


# In[109]:


    et_train=et_hpo.predict(X_train)
    et_test=et_hpo.predict(X_test)


# ### Apply Stacking
# The ensemble model that combines the four ML models (DT, RF, ET, XGBoost)

# In[116]:


    base_predictions_train = pd.DataFrame( {
        'DecisionTree': dt_train.ravel(),
            'RandomForest': rf_train.ravel(),
         'ExtraTrees': et_train.ravel(),
         'XgBoost': xg_train.ravel(),
        })
    base_predictions_train.head(5)


# In[117]:


    dt_train=dt_train.reshape(-1, 1)
    et_train=et_train.reshape(-1, 1)
    rf_train=rf_train.reshape(-1, 1)
    xg_train=xg_train.reshape(-1, 1)
    dt_test=dt_test.reshape(-1, 1)
    et_test=et_test.reshape(-1, 1)
    rf_test=rf_test.reshape(-1, 1)
    xg_test=xg_test.reshape(-1, 1)


# In[118]:


    dt_train.shape


# In[119]:


    x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
    x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)


# In[120]:


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
    # 


# In[124]:


    xg = xgb.XGBClassifier(learning_rate= 0.19229249758051492, n_estimators = 30, max_depth = 36)
    xg.fit(x_train,y_train)
    xg_score=xg.score(x_test,y_test)
    y_predict=xg.predict(x_test)
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
    # 


# ## Anomaly-based IDS

# ### Generate the port-scan datasets for unknown attack detection

# In[131]:


    df=pd.read_csv('./data/CICIDS2017_sample_km.csv')


# In[132]:


    df.Label.value_counts()


# In[144]:


    df1 = df[df['Label'] != 5]
    df1['Label'][df1['Label'] > 0] = 1
    df1.to_csv('./data/CICIDS2017_sample_km_without_portscan.csv',index=0)


# In[145]:


    df2 = df[df['Label'] == 5]
    df2['Label'][df2['Label'] == 5] = 1
    df2.to_csv('./data/CICIDS2017_sample_km_portscan.csv',index=0)


# ### Read the generated datasets for unknown attack detection

# In[67]:


    df1 = pd.read_csv('./data/CICIDS2017_sample_km_without_portscan.csv')
    df2 = pd.read_csv('./data/CICIDS2017_sample_km_portscan.csv')


# In[68]:


    features = df1.drop(['Label'],axis=1).dtypes[df1.dtypes != 'object'].index
    df1[features] = df1[features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    df2[features] = df2[features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)


# In[69]:


    df1.Label.value_counts()


# In[70]:


    df2.Label.value_counts()


# In[71]:


    df2p=df1[df1['Label']==0]
    df2pp=df2p.sample(n=None, frac=1255/18225, replace=False, weights=None, random_state=None, axis=0)
    df2=pd.concat([df2, df2pp])


# In[72]:


    df2.Label.value_counts()


# In[73]:


    df = df1._append(df2)


# In[74]:


    X = df.drop(['Label'],axis=1) .values
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)
    pd.Series(y).value_counts()


# ### Feature engineering (IG, FCBF, and KPCA)

# #### Feature selection by information gain (IG)

# In[75]:


    from sklearn.feature_selection import mutual_info_classif
    importances = mutual_info_classif(X, y)


# In[76]:


# calculate the sum of importance scores
    f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])


# In[77]:


# select the important features from top to bottom until the accumulated importance reaches 90%
    f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
    Sum2 = 0
    fs = []
    for i in range(0, len(f_list2)):
        Sum2 = Sum2 + f_list2[i][0]
        fs.append(f_list2[i][1])
        if Sum2>=0.9:
            break        


# In[78]:


    X_fs = df[fs].values


# In[79]:


    X_fs.shape


# In[80]:


    X_fs


# #### Feature selection by Fast Correlation Based Filter (FCBF)
# 
# The module is imported from the GitHub repo: https://github.com/SantiagoEG/FCBF_module

# In[81]:


    from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
    fcbf = FCBFK(k = 20)
#fcbf.fit(X_fs, y)


# In[82]:


    X_fss = fcbf.fit_transform(X_fs,y)


# In[83]:


    X_fss.shape


# In[84]:


    X_fss


# ####  kernel principal component analysis (KPCA)

# In[123]:


    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components = 10, kernel = 'rbf')
    kpca.fit(X_fss, y)
    X_kpca = kpca.transform(X_fss)

# from sklearn.decomposition import PCA
# kpca = PCA(n_components = 10)
# kpca.fit(X_fss, y)
# X_kpca = kpca.transform(X_fss)


# ### Train-test split after feature selection

# In[86]:


    X_train = X_kpca[:len(df1)]
    y_train = y[:len(df1)]
    X_test = X_kpca[len(df1):]
    y_test = y[len(df1):]


# ### Solve class-imbalance by SMOTE

# In[87]:


    pd.Series(y_train).value_counts()


# In[88]:


    from imblearn.over_sampling import SMOTE
    smote=SMOTE(n_jobs=-1,sampling_strategy={1:18225})
    X_train, y_train = smote.fit_resample(X_train, y_train)


# In[89]:


    pd.Series(y_train).value_counts()


# In[90]:


    pd.Series(y_test).value_counts()


# ### Apply the cluster labeling (CL) k-means method

# In[91]:


    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN,MeanShift
    from sklearn.cluster import SpectralClustering,AgglomerativeClustering,AffinityPropagation,Birch,MiniBatchKMeans,MeanShift 
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    from sklearn.metrics import classification_report
    from sklearn import metrics


# In[124]:


    def CL_kmeans(X_train, X_test, y_train, y_test,n,b=100):
        km_cluster = MiniBatchKMeans(n_clusters=n,batch_size=b)
        result = km_cluster.fit_predict(X_train)
        result2 = km_cluster.predict(X_test)

        count=0
        a=np.zeros(n)
        b=np.zeros(n)
        for v in range(0,n):
            for i in range(0,len(y_train)):
                if result[i]==v:
                    if y_train[i]==1:
                        a[v]=a[v]+1
                    else:
                        b[v]=b[v]+1
        list1=[]
        list2=[]
        for v in range(0,n):
            if a[v]<=b[v]:
                list1.append(v)
            else: 
                list2.append(v)
        for v in range(0,len(y_test)):
            if result2[v] in list1:
                result2[v]=0
            elif result2[v] in list2:
                result2[v]=1
            else:
                print("-1")
        print(classification_report(y_test, result2))
        cm=confusion_matrix(y_test,result2)
        acc=metrics.accuracy_score(y_test,result2)
        print(str(acc))
        print(cm)


# In[94]:


    CL_kmeans(X_train, X_test, y_train, y_test, 8)

    CL_kmeans(X_train, X_test, y_train, y_test, 16)


# ### Apply the CL-k-means model with biased classifiers

# In[125]:


# Only a sample code to show the logic. It needs to work on the entire dataset to generate sufficient training samples for biased classifiers
    def Anomaly_IDS(X_train, X_test, y_train, y_test,n,b=100):
        # CL-kmeans
        km_cluster = MiniBatchKMeans(n_clusters=n,batch_size=b)
        result = km_cluster.fit_predict(X_train)
        result2 = km_cluster.predict(X_test)

        count=0
        a=np.zeros(n)
        b=np.zeros(n)
        for v in range(0,n):
            for i in range(0,len(y_train)):
                if result[i]==v:
                    if y_train[i]==1:
                        a[v]=a[v]+1
                    else:
                        b[v]=b[v]+1
        list1=[]
        list2=[]
        for v in range(0,n):
            if a[v]<=b[v]:
                list1.append(v)
            else: 
                list2.append(v)
        for v in range(0,len(y_test)):
            if result2[v] in list1:
                result2[v]=0
            elif result2[v] in list2:
                result2[v]=1
            else:
                print("-1")
        print(classification_report(y_test, result2))
        cm=confusion_matrix(y_test,result2)
        acc=metrics.accuracy_score(y2,result2)
        print(str(acc))
        print(cm)
        
        #Biased classifier construction
        count=0
        print(len(y))
        a=np.zeros(n)
        b=np.zeros(n)
        FNL=[]
        FPL=[]
        for v in range(0,n):
            al=[]
            bl=[]
            for i in range(0,len(y)):   
                if result[i]==v:        
                    if y[i]==1:        #label 1
                        a[v]=a[v]+1
                        al.append(i)
                    else:             #label 0
                        b[v]=b[v]+1
                        bl.append(i)
            if a[v]<=b[v]:
                FNL.extend(al)
            else:
                FPL.extend(bl)
            #print(str(v)+"="+str(a[v]/(a[v]+b[v])))
            
        dffp=df.iloc[FPL, :]
        dffn=df.iloc[FNL, :]
        dfva0=df[df['Label']==0]
        dfva1=df[df['Label']==1]
        
        dffpp=dfva1.sample(n=None, frac=len(FPL)/dfva1.shape[0], replace=False, weights=None, random_state=None, axis=0)
        dffnp=dfva0.sample(n=None, frac=len(FNL)/dfva0.shape[0], replace=False, weights=None, random_state=None, axis=0)
        
        dffp_f=pd.concat([dffp, dffpp])
        dffn_f=pd.concat([dffn, dffnp])
        
        Xp = dffp_f.drop(['Label'],axis=1)  
        yp = dffp_f.iloc[:, -1].values.reshape(-1,1)
        yp=np.ravel(yp)

        Xn = dffn_f.drop(['Label'],axis=1)  
        yn = dffn_f.iloc[:, -1].values.reshape(-1,1)
        yn=np.ravel(yn)
        
        rfp = RandomForestClassifier(random_state = 0)
        rfp.fit(Xp,yp)
        rfn = RandomForestClassifier(random_state = 0)
        rfn.fit(Xn,yn)

        dffnn_f=pd.concat([dffn, dffnp])
        
        Xnn = dffn_f.drop(['Label'],axis=1)  
        ynn = dffn_f.iloc[:, -1].values.reshape(-1,1)
        ynn=np.ravel(ynn)

        rfnn = RandomForestClassifier(random_state = 0)
        rfnn.fit(Xnn,ynn)

        X2p = df2.drop(['Label'],axis=1) 
        y2p = df2.iloc[:, -1].values.reshape(-1,1)
        y2p=np.ravel(y2p)

        result2 = km_cluster.predict(X2p)

        count=0
        a=np.zeros(n)
        b=np.zeros(n)
        for v in range(0,n):
            for i in range(0,len(y)):
                if result[i]==v:
                    if y[i]==1:
                        a[v]=a[v]+1
                    else:
                        b[v]=b[v]+1
        list1=[]
        list2=[]
        l1=[]
        l0=[]
        for v in range(0,n):
            if a[v]<=b[v]:
                list1.append(v)
            else: 
                list2.append(v)
        for v in range(0,len(y2p)):
            if result2[v] in list1:
                result2[v]=0
                l0.append(v)
            elif result2[v] in list2:
                result2[v]=1
                l1.append(v)
            else:
                print("-1")
        print(classification_report(y2p, result2))
        cm=confusion_matrix(y2p,result2)
        print(cm)

    return {}

# 95% of the code has been shared, and the remaining 5% is retained for future extension.  
# Thank you for your interest and more details are in the paper.
