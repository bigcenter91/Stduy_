import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.utils import all_estimators
warnings.filterwarnings(action='ignore')
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline,Pipeline

# 1. 데이터
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)]

data_name_list = ['iris : ',
                  'breast_cancer :',
                  'digits :',
                  'load_wine']

scaler_list = [MinMaxScaler(),
               StandardScaler(),
               MaxAbsScaler(),
               RobustScaler()]

model_list = [SVC(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]

scaler_name_list = ['MinMaxScaler',
                    'StandardScaler',
                    'MaxAbsScaler',
                    'RobustScaler']

model_name_list = ['SVC',
                   'DecisionTreeClassifier',
                   'RandomForestClassifier']

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=413)

max_score = 0
max_name = '최대값'
max_scaler = 'max'

for index, value in enumerate(data_list):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=27)
    
    for i, value2 in enumerate(scaler_list):
        scaler = value2
        model_score_list = []
        
        for j, value3 in enumerate(model_list):
            model = value3
            pipeline = Pipeline([("sca",scaler),( "mod",model)])
            scores = cross_val_score(pipeline, x_train, y_train, cv=kfold)
            model_score = np.mean(scores)
            model_score_list.append(model_score)
            
            if model_score > max_score:
                max_score = model_score
                max_name = model_name_list[j]
                max_scaler = scaler_name_list[i]

        print(f"\nData: {data_name_list[index]}, Scaler: {scaler_name_list[i]}")
        for k, model_score in enumerate(model_score_list):
            print(f"{model_name_list[k]}: {model_score:.4f}")
        print(f"Data: {data_name_list[index]} Best Model: {max_name} with a score of {max_score:.4f} using {max_scaler} scaling")