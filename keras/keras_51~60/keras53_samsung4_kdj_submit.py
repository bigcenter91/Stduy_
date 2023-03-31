import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler,Normalizer
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))

def split_x(dt, st):
    a = []
    for i in range(len(dt)-st-1):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/시험/'
path_save = './_save/samsung/'

datasets_s = pd.read_csv(path + '삼성전자 주가3.csv', index_col=0, encoding='EUC-KR')
datasets_h = pd.read_csv(path + '현대자동차2.csv', index_col=0, encoding='EUC-KR')

print(datasets_s.shape, datasets_h.shape)
print(datasets_s.columns, datasets_h.columns)
print(datasets_s.info(), datasets_h.info())
print(datasets_s.describe(), datasets_h.describe())
print(type(datasets_s), type(datasets_h))

samsung_x = np.array(datasets_s.drop(['전일비', '시가'], axis=1))
samsung_y = np.array(datasets_s['시가'])
hyundai_x = np.array(datasets_h.drop(['전일비', '시가'], axis=1))
hyundai_y = np.array(datasets_h['시가'])

samsung_x = samsung_x[:200, :]
samsung_y = samsung_y[:200]
hyundai_x = hyundai_x[:200, :]
hyundai_y = hyundai_y[:200]

samsung_x = np.flip(samsung_x, axis=1)
samsung_y = np.flip(samsung_y)
hyundai_x = np.flip(hyundai_x, axis=1)
hyundai_y = np.flip(hyundai_y)

print(samsung_x.shape, samsung_y.shape) # (200, 13) (200,)
print(hyundai_x.shape, hyundai_y.shape) # (200, 13) (200,)

samsung_x = np.char.replace(samsung_x.astype(str), ',', '').astype(np.float64)
samsung_y = np.char.replace(samsung_y.astype(str), ',', '').astype(np.float64)
hyundai_x = np.char.replace(hyundai_x.astype(str), ',', '').astype(np.float64)
hyundai_y = np.char.replace(hyundai_y.astype(str), ',', '').astype(np.float64)

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, hyundai_x_train, hyundai_x_test, hyundai_y_train, hyundai_y_test = train_test_split(
    samsung_x, samsung_y, hyundai_x, hyundai_y, train_size=0.7, shuffle=False)

scaler = MinMaxScaler()
samsung_x_train = scaler.fit_transform(samsung_x_train)
samsung_x_test= scaler.transform(samsung_x_test)
hyundai_x_train = scaler.transform(hyundai_x_train)
hyundai_x_test = scaler.transform(hyundai_x_test)

timesteps = 10

samsung_x_train_split = split_x(samsung_x_train, timesteps)
samsung_x_test_split = split_x(samsung_x_test, timesteps)
hyundai_x_train_split = split_x(hyundai_x_train, timesteps)
hyundai_x_test_split = split_x(hyundai_x_test, timesteps)

samsung_y_train_split = samsung_y_train[(timesteps+1):]
samsung_y_test_split = samsung_y_test[(timesteps+1):]
hyundai_y_train_split = hyundai_y_train[(timesteps+1):]
hyundai_y_test_split = hyundai_y_test[(timesteps+1):]

# samsung_y_test_split = np.roll(samsung_y_test_split, -2)
# hyundai_y_test_split = np.roll(hyundai_y_test_split, -2)

print(samsung_x_train_split.shape)      # (119, 20, 13)
print(hyundai_x_train_split.shape)      # (119, 20, 13)

from tensorflow.keras.optimizers import Adam
class CustomAdam(Adam):
    pass


optimizer = Adam()


model = load_model('./_save/samsung/adddaa.h5')

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


#4. 평가, 예측
loss = model.evaluate([samsung_x_test_split, hyundai_x_test_split], [samsung_y_test_split, hyundai_y_test_split])
print('loss : ', loss)

samsung_x_predict = samsung_x_test[-timesteps:]
samsung_x_predict = samsung_x_predict.reshape(1, timesteps, 14)
hyundai_x_predict = hyundai_x_test[-timesteps:]
hyundai_x_predict = hyundai_x_predict.reshape(1, timesteps, 14)

predict_result = model.predict([samsung_x_predict, hyundai_x_predict])

print("시가는 : ", np.round(predict_result[1], 2))