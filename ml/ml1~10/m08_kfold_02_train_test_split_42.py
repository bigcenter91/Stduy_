import numpy as np
from sklearn.datasets import load_boston,fetch_california_housing,load_diabetes
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, StratifiedKFold
import warnings
from sklearn.preprocessing import RobustScaler,StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn.utils import all_estimators
warnings.filterwarnings(action = 'ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
data_list = [load_boston(return_X_y=True),
             fetch_california_housing(return_X_y=True),
             load_diabetes(return_X_y=True)]

data_name_list = ['boston',
                  'california',
                  'diabetes']

scaler_list = [MinMaxScaler(),
               StandardScaler(),
               MaxAbsScaler(),
               RobustScaler()]

scaler_name_list = ['MinMaxScaler',
                    'StandardScaler',
                    'MaxAbsScaler',
                    'RobustScaler']

n_splits=5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 413)

#1. 데이터
for index, value in enumerate(data_list):
    x, y = value
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1534, train_size=0.7, shuffle=True)
    print("==================", data_name_list[index], "======================")
    for i, scaler in enumerate(scaler_list):
        scaler_name = scaler_name_list[i]
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        max_score = 0
        max_name = '최대값'
        max_scaler = 'max'
        for name, algorithm in all_estimators(type_filter='regressor'):
            try:
                model = algorithm()
                scores = cross_val_score(model, x_train_scaled, y_train, cv=kfold, scoring='r2')
                if max_score < scores.mean():
                    max_score = scores.mean()
                    max_name = name
                    max_scaler = scaler_name_list[i]
            except:
                continue
        print('Scaler:', max_scaler, 'Model:', max_name, 'Score:', max_score)
    print("======================================================")



