import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.utils import all_estimators

warnings.filterwarnings(action='ignore')

# 1. 데이터
data_list = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_digits(return_X_y=True),
    load_wine(return_X_y=True)
]

data_name_list = [
    'iris',
    'breast_cancer',
    'digits',
    'wine'
]

scaler_list = [
    MinMaxScaler(),
    StandardScaler(),
    MaxAbsScaler(),
    RobustScaler()
]

scaler_name_list = [
    'MinMaxScaler',
    'StandardScaler',
    'MaxAbsScaler',
    'RobustScaler'
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=413)
allAlgorithms = all_estimators(type_filter='classifier')

# 2. 모델구성
max_r2 = np.zeros(len(allAlgorithms))
max_name = ['']*len(allAlgorithms)

for i, (x, y) in enumerate(data_list):
    for j, scaler in enumerate(scaler_list):
        print("="*30)
        print(scaler_name_list[j])
        print("="*30)
        print(data_name_list[i])
        for name, algorithm in allAlgorithms:
            try:  # 예외 처리
                model = algorithm()
                print("-"*50)
                # 3. 훈련
                print(name)
                # 4. 평가, 예측
                scores = cross_val_score(model, scaler.fit_transform(x), y, cv=kfold)
                print('ACC:', scores)
                print('cross_val_score average:', round(np.mean(scores), 4))

                if max_r2[name] < np.mean(scores):
                    max_r2[name] = np.mean(scores)  # early stopping
                    max_name[name] = name.__name__

            except Exception as e:
                print(algorithm.__name__, "에러:", e)

print("="*50)
print('최고모델:', max_name[np.argmax(max_r2)], max_r2[np.argmax(max_r2)])
print("="*50)
