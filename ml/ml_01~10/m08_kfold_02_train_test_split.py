import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators # 모든 모델이 들어가 있다.
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#1 데이터
ddarung_path = 'c:/_study/_data/_ddarung/'
kaggle_bike_path = 'c:/_study/_data/_kaggle_bike/'

ddarung_train = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike_train = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()


data_list = data_list = [fetch_california_housing(return_X_y = True),
             load_diabetes(return_X_y = True),
             ddarung_train,
             kaggle_bike_train]

data_list_name = ['캘리포니아',
                  '디아뱃',
                  '따릉이',
                  '캐글 바이크']

scaler = StandardScaler()

n_split = 5

kfold = KFold(n_splits = n_split, shuffle = True, random_state = 123)

for i in range(len(data_list)):
    if i < 2:
        x, y = data_list[i]
    elif i == 2:
        x = data_list[i].drop(['count'], axis = 1)
        y = data_list[i]['count']
    # elif i == 3:
    else:
        x = data_list[i].drop(['count', 'registered', 'casual'], axis = 1)
        y = data_list[i]['count']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 123)
    # x = scaler.fit_transform(x)

    allAlgorithms = all_estimators(type_filter = 'classifier')

    max_score = 0
    max_name = 'ㅇ'

    for (name, algorithms) in allAlgorithms:
        try:
            model = algorithms()
            
            scores = cross_val_score(model, x_train, y_train, cv = kfold)
            results = round(np.mean(scores), 4)
            
            y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
            acc = accuracy_score(y_test, y_predict)
            
            if max_score < results:
               max_score = results
               max_score = name
            
        except:
            continue
    
print('acc : ', acc)
print("==============", data_list_name[i], "===============")
print("최고모델 : ", max_name, max_score)
print("==================================================")