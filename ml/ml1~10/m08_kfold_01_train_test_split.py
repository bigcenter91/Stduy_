import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,load_wine
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, StratifiedKFold
import warnings
from sklearn.preprocessing import RobustScaler,StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn.utils import all_estimators
warnings.filterwarnings(action = 'ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#1. 데이터

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)
             ,]

data_name_list = ['iris',
                  'breast_cancer',
                  'digits',
                  'wine',
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
kfold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 413)

max_score = 0
max_name = '최대값'
max_scaler = 'max'

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
        for name, algorithm in all_estimators(type_filter='classifier'):
            try:
                model = algorithm()
                y_predict = cross_val_predict(model, x_test_scaled, y_test, cv= kfold)
                acc = accuracy_score(y_test, y_predict)
                results =round(np.mean(acc),4)
                if max_score < results:
                    max_score = results
                    max_name = name
                    max_scaler = scaler_name_list[i]
            except:
                continue
        print('Scaler:', max_scaler, 'Model:', max_name, 'Score:', max_score)
    print("======================================================")


# ================== iris ======================
# Scaler: MinMaxScaler Model: LinearDiscriminantAnalysis Score: 1.0
# Scaler: StandardScaler Model: LinearDiscriminantAnalysis Score: 1.0
# Scaler: MaxAbsScaler Model: LinearDiscriminantAnalysis Score: 1.0
# Scaler: RobustScaler Model: LinearDiscriminantAnalysis Score: 1.0
# ======================================================
# ================== breast_cancer ======================
# Scaler: MinMaxScaler Model: LogisticRegression Score: 0.9415
# Scaler: StandardScaler Model: ExtraTreesClassifier Score: 0.9415
# Scaler: MaxAbsScaler Model: ExtraTreesClassifier Score: 0.9474
# Scaler: RobustScaler Model: ExtraTreesClassifier Score: 0.9474
# ======================================================
# ================== digits ======================
# Scaler: MinMaxScaler Model: KNeighborsClassifier Score: 0.963
# Scaler: StandardScaler Model: ExtraTreesClassifier Score: 0.9667
# Scaler: MaxAbsScaler Model: KNeighborsClassifier Score: 0.963
# Scaler: RobustScaler Model: KNeighborsClassifier Score: 0.963
# ======================================================
# ================== wine ======================
# Scaler: MinMaxScaler Model: ExtraTreesClassifier Score: 0.9815
# Scaler: StandardScaler Model: ExtraTreesClassifier Score: 0.9815
# Scaler: MaxAbsScaler Model: ExtraTreesClassifier Score: 1.0
# Scaler: RobustScaler Model: ExtraTreesClassifier Score: 0.9815
# ======================================================
