import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tqdm import tqdm

# 1 데이터
dacon_diabets_path = './_data/dacon_diabetes/'
ddarung_path = './_data/ddarung/'
kaggle_bike_path = './_data/kaggle_bike/'

dacon_diabets = pd.read_csv(dacon_diabets_path + 'train.csv', index_col = 0).dropna()
ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

x1 = dacon_diabets.drop(['Outcome'], axis = 1)
y1 = dacon_diabets['Outcome']

x2 = ddarung.drop(['count'], axis = 1)
y2 = ddarung['count']

x3 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1)
y3 = kaggle_bike['count']

data_list = {'iris' : load_iris,
             'cancer' : load_breast_cancer,
             'wine' : load_wine,
             'digits' : load_digits,
             'diabetes' : load_diabetes,
             'california' : fetch_california_housing,
             'dacon_diabets' : (x1, y1),
             'ddarung' : (x2, y2),
             'kaggle_bike' : (x3, y3)}

scaler_list = {'MinMax' : MinMaxScaler,
               'MaxAbs' : MaxAbsScaler,
               'Standard' : StandardScaler,
               'Robust' : RobustScaler}

total_datasets = len(data_list)
total_scalers = len(scaler_list)

for i, d in enumerate(data_list): # tqdm 현재 코드의 실행 상태를 터미널창에 퍼센트로 알려줌
    print(f'데이터 진행 상태: {d} ({i/total_datasets * 100:.1f}%)')
    if d == 'iris' or d == 'cancer' or d == 'wine' or d == 'digits' or d == 'diabetes' or d == 'california':
        x, y = data_list[d](return_X_y = True)
    elif d == 'dacon_diabets' or d == 'ddarung' or d == 'kaggle_bike':
        x, y = data_list[d]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    
    for j, s in enumerate(scaler_list): # tqdm 현재 코드의 실행 상태를 터미널창에 퍼센트로 알려줌
        print(f'스케일러 진행 상태: {d} ({j/total_datasets * 100:.1f}%)')
        scaler = scaler_list[s]()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    
        #2 모델 구성
        model = Sequential()
        model.add(Dense(64, input_dim = x_train.shape[1]))
        model.add(Dense(32))
        model.add(Dense(32))
        model.add(Dense(32))
        model.add(Dense(1, activation = 'sigmoid'))
        
        # 3 컴파일, 훈련
        learning_rate = 0.1
        
        optimizer = Adam(learning_rate = learning_rate)
        
        model.compile(loss = 'mse', optimizer = optimizer)
        
        es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min', verbose = 1)
        
        rlr = ReduceLROnPlateau(monitor = 'val_loss', patience = 30, mode = 'auto', verbose = 1)
        
        model.fit(x_train, y_train, epochs = 3, batch_size = 32, validation_split = 0.2, verbose = 1, callbacks = [es, rlr])
        
        # 4 평가, 예측
        loss = model.evaluate(x_test, y_test)
        print(f'데이터 : {d}, 스케일러 : {s}, 손실값 : {loss}')
