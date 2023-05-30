import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,fetch_covtype,load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler,StandardScaler,MaxAbsScaler,MinMaxScaler
warnings.filterwarnings(action = 'ignore')

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)
             ,fetch_covtype(return_X_y=True),
             ]

model_list = [LinearSVC(),
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]

data_name_list = ['iris : ',
                  'breast_cancer :',
                  'digits :',
                  'load_wine',
                  'covtype']

model_name_list = ['LinearSVC',
              'LogisticRegression',
              'DecisionTreeClassifier',
              'RandomForestClassifier']

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

#2. 모델
for  i, value in enumerate(data_list): #enumerate수치와 순서를 나타내주는 함수.
    x, y = value #첫번째 iris들어가고 두번째 cancer가 들어감
    for a, value in enumerate(scaler_list):
        print("=============================")
        print(scaler_name_list[a])
        print("=============================")
        print(data_name_list[i])
        for j, value2 in enumerate(model_list):
            model = value2 #j가 들어가면 데이터값이 나와버림

            #4. 평가, 예측
            scores = cross_val_score(model, x, y, cv = kfold)
            print(model_name_list[j], 'ACC :' ,scores,
                '\n cross_val_score average : ', round(np.mean(scores),4))
            
# =============================
# MinMaxScaler
# =============================
# iris : 
# LinearSVC ACC : [0.86666667 0.93333333 1.         1.         0.86666667] 
#  cross_val_score average :  0.9333
# LogisticRegression ACC : [0.86666667 0.93333333 1.         0.96666667 0.9       ]     
#  cross_val_score average :  0.9333
# DecisionTreeClassifier ACC : [0.9        0.93333333 1.         0.93333333 0.93333333] 
#  cross_val_score average :  0.94
# RandomForestClassifier ACC : [0.9        0.93333333 1.         1.         0.96666667] 
#  cross_val_score average :  0.96