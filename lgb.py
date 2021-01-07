import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score

pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.max_rows', None) # 显示所有行
train_data = pd.read_csv("F:\\ML\\train.csv",dtype={ "GENDER": object})
test_data = pd.read_csv("F:\\ML\\test.csv",dtype={ "GENDER": object})

drop_columns = []
drop_columns.append('CUST_ID')
train_data.groupby('IDF_TYP_CD')['IDF_TYP_CD'].count()
drop_columns.append('IDF_TYP_CD')
orig_columns=test_data.columns
for col in orig_columns:
    col_series = train_data[col].dropna().unique()
    if len(col_series)==1:
        drop_columns.append(col)
testLa = test_data['CUST_ID']
train_data=train_data.drop(drop_columns,axis=1)
test_data=test_data.drop(drop_columns,axis=1)
object_columns_df = train_data.select_dtypes(include=['object'])
object_columns = object_columns_df.columns
gender_replace = {'GENDER':{'X':0}}
train_data = train_data.replace(gender_replace)
test_data = test_data.replace(gender_replace)
train_data['GENDER'] = train_data['GENDER'].astype('int')
test_data['GENDER'] = test_data['GENDER'].astype('int')
gender_replace = {'C_FUND_FLAG':{'N':-1},'D_FUND_FLAG':{'N':-1},'S_FUND_FLAG':{'N':-1}}
train_data = train_data.replace(gender_replace)
test_data = test_data.replace(gender_replace)
train_data['C_FUND_FLAG'] = train_data['C_FUND_FLAG'].astype('int')
test_data['C_FUND_FLAG'] = test_data['C_FUND_FLAG'].astype('int')
train_data['D_FUND_FLAG'] = train_data['D_FUND_FLAG'].astype('int')
test_data['D_FUND_FLAG'] = test_data['D_FUND_FLAG'].astype('int')
train_data['S_FUND_FLAG'] = train_data['S_FUND_FLAG'].astype('int')
test_data['S_FUND_FLAG'] = test_data['S_FUND_FLAG'].astype('int')
for col in object_columns[1:]:
    if col!='C_FUND_FLAG' and col!='D_FUND_FLAG' and col!='S_FUND_FLAG':
        flag_replace={col:{'Y':1,'N':0}}
        train_data = train_data.replace(flag_replace)
        test_data = test_data.replace(flag_replace)
        train_data[col] = train_data[col].astype('int')
        test_data[col] = test_data[col].astype('int')
X_pre = test_data
X = train_data.drop(['bad_good'],axis=1)
Y = train_data['bad_good']

模型训练过程
scaler = MinMaxScaler()
X_sca = scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(X_sca,Y,test_size=0.2,random_state=0)
oversampler=SMOTE(random_state=0)
os_x_train,os_y_train=oversampler.fit_sample(x_train,y_train)
pca_lr = Pipeline([('pca',PCA(n_components='mle')),('lr',LogisticRegression(random_state=1,solver='sag'))])
parameters = {'lr__C':[0.001,0.01,0.1]}
clf = GridSearchCV(pca_lr,parameters,cv=3,scoring='f1_macro')
clf.fit(os_x_train,os_y_train)
print(clf.error_score)
