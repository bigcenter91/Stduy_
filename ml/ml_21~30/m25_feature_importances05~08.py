import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

#1 데이터
ddarung_path = 'c:/stduy_/_data/ddarung/'
kaggle_bike_path = 'c:/stduy_/_data/kaggle_bike/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()



x1 = ddarung.drop(['count'], axis = 1)
y1 = ddarung['count']

x2 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1)
y2 = kaggle_bike['count']

data_list = [fetch_california_housing(return_X_y = True),
             load_diabetes(return_X_y = True),
             (x1, y1),
             (x2, y2)]

data_list_name = ['캘리포니아',
                  '당뇨병',
                  '따릉이',
                  '캐글 바이크']

for i in range(len(data_list)):
    x, y = data_list[i]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 123)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #2 모델
    model = RandomForestRegressor()
    
    #3 컴파일, 훈련
    model.fit(x_train, y_train)
    
    #4 평가, 예측
    result = model.score(x_test, y_test)
    print(data_list_name[i], 'result : ', result)
    
    y_predict = model.predict(x_test)
    
    r2 = r2_score(y_test, y_predict)
    print(data_list_name[i], 'r2 : ', r2)
    print(model, ':', model.feature_importances_)
    
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 123)
    
    model.fit(x_train1, y_train1)
    
    result = model.score(x_test1, y_test1)
    print(data_list_name[i], 'result : ', result)
    
    y_predict1 = model.predict(x_test1)
    
    r22 = r2_score(y_test1, y_predict1)
    print(data_list_name[i], 'acc1 : ', r22)