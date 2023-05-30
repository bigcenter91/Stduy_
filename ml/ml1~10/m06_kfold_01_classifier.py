import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,load_wine
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler,StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn.utils import all_estimators
warnings.filterwarnings(action = 'ignore')
#1. 데이터

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)
             ,]

data_name_list = ['iris : ',
                  'breast_cancer :',
                  'digits :',
                  'load_wine',
                  ]

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

max_score = 0
max_name = '최대값'
max_scaler = 'max'
#1. 데이터
for index, value in enumerate(data_list):
    x, y = value 
    for i, value2 in enumerate(scaler_list):
        scaler = value2
        x_scaler = scaler.fit_transform(x)
        #2. 모델구성
        allAlgorithms = all_estimators(type_filter='classifier')
        max_score = 0
        max_name = '최대값'
        max_scaler = 'max'
        for (name , algorithm) in allAlgorithms:
                
                try: #예외 처리
                    model =  algorithm()
                        
                    scores = cross_val_score(model, x_scaler, y, cv = kfold)
                    results =round(np.mean(scores),4)
                                
                    if max_score < results :
                        max_score = results #earlystopping이랑 비슷함(계속 최댓값만 저장.)
                        max_name = name
                        max_scaler = name
                except:
                    continue

print("==============" ,data_name_list[index], "======================")
print('최고모델 :', max_scaler, max_score)
print("================================================")

# ============== load_wine ======================
# 최고모델 : RandomForestClassifier RandomForestClassifier 0.9889