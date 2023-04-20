import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing, load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

#1 데이터
ddarung_path = './_data/ddarung/'
kaggle_bike_path = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike_train = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

data_list = [load_boston(return_X_y = True),
             fetch_california_housing(return_X_y = True),
             ddarung_train,
             kaggle_bike_train]

data_list_name = ['보스턴',
                  '캘리포니아',
                  '따릉이',
                  '캐글 바이크']

scaler = MinMaxScaler()

n_split = 5
kfold = KFold(n_splits = n_split, shuffle = True, random_state = 123)

for i, v in enumerate(data_list):
    
    x, y = v
    x = scaler.fit_transform(x)

    allAlgorithms = all_estimators(type_filter = 'regressor')

    max_score = 0
    max_name = '바보'

    for (name, algorithms) in allAlgorithms:
        try:
            model = algorithms()
            
            scores = cross_val_score(model, x, y, cv = kfold)
            results = round(np.mean(scores), 4)
            
            if max_score < results:
               max_score = results
               max_score = name
            
        except:
            continue
    
print("==============", data_list_name[i], "===============")
print("최고모델 : ", max_name, max_score)
print("==================================================")