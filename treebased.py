import warnings 
warnings.filterwarnings("ignore")
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


df = pd.read_csv('./data/CICIDS2017_sample_km.csv')

print(df)

print(df.Label.value_counts())

# split train set and test set
X = df.drop(['Label'],axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0) #shuffle=False
print(X_train, X_test, y_train, y_test)

## SMOTE to solve class imbalance

pd.Series(y_train).value_counts()

smote=SMOTE(n_jobs=-1, sampling_strategy={2:1000,4:1000})
X_train, y_train = smote.fit_resample(X_train, y_train)

print(X_train, y_train)

## Training three base learners lightgbm, xgboost, catboost

# %%time
# Train the LightGBM algorithm
import lightgbm as lgb
lg = lgb.LGBMClassifier()
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
print(classification_report(y_test,y_pred))
print('Accuracy of LightGBM: ' + str(accuracy_score(y_test, y_pred)))
print('Precision of LightGBM: ' + str(precision_score(y_test, y_pred, average='weighted')))
print('Recall of LightGBM: ' + str(recall_score(y_test, y_pred, average='weighted')))
print('Average F1 of LightGBM: ' + str(f1_score(y_test, y_pred, average='weighted')))
print('F1 of LightGBM for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
lg_f1=f1_score(y_test, y_pred, average=None)

# Plot the confusion matrix
cm=confusion_matrix(y_test,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor='red',fmt='.0f',ax=ax)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.title('lightgbm confusion matrix')
plt.savefig('lightgbm_confusion_matrix.png')


# Train the XGBoost algorithm
import xgboost as xgb
xg = xgb.XGBClassifier()

X_train_x = X_train.values
X_test_x = X_test.values

xg.fit(X_train_x, y_train)

y_pred = xg.predict(X_test_x)
print(classification_report(y_test,y_pred))
print('Accuracy of XGBoost: '+ str(accuracy_score(y_test, y_pred)))
print('Precision of XGBoost: '+ str(precision_score(y_test, y_pred, average='weighted')))
print('Recall of XGBoost: '+ str(recall_score(y_test, y_pred, average='weighted')))
print('Average F1 of XGBoost: '+ str(f1_score(y_test, y_pred, average='weighted')))
print('F1 of XGBoost for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
xg_f1=f1_score(y_test, y_pred, average=None)

# Plot the confusion matrix
cm=confusion_matrix(y_test,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor='red',fmt='.0f',ax=ax)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.title('xgboost confusion matrix')
plt.savefig('xgboost_confusion_matrix.png')


# Train the CatBoost algorithm
import catboost as cbt
cb = cbt.CatBoostClassifier(verbose=0,boosting_type='Plain')
#cb = cbt.CatBoostClassifier()

cb.fit(X_train, y_train)
y_pred = cb.predict(X_test)
print(classification_report(y_test,y_pred))
print('Accuracy of CatBoost: '+ str(accuracy_score(y_test, y_pred)))
print('Precision of CatBoost: '+ str(precision_score(y_test, y_pred, average='weighted')))
print('Recall of CatBoost: '+ str(recall_score(y_test, y_pred, average='weighted')))
print('Average F1 of CatBoost: '+ str(f1_score(y_test, y_pred, average='weighted')))
print('F1 of CatBoost for each type of attack: '+ str(f1_score(y_test, y_pred, average=None)))
cb_f1=f1_score(y_test, y_pred, average=None)

# Plot the confusion matrix
cm=confusion_matrix(y_test,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor='red',fmt='.0f',ax=ax)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.title('catboost confusion matrix')
plt.savefig('catboost_confusion_matrix.png')
