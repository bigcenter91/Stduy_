import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))

def split_x(dt, st):
    a = []
    for i in range(len(dt)-st):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/시험/'
path_save = './_save/samsung/'

datasets_samsung = pd.read_csv(path + '삼성전자 주가2.csv', index_col=0, encoding='cp949')
datasets_hyundai = pd.read_csv(path + '현대자동차.csv', index_col=0, encoding='cp949')

print(datasets_samsung.shape, datasets_hyundai.shape)
print(datasets_samsung.columns, datasets_hyundai.columns)
print(datasets_samsung.info(), datasets_hyundai.info())
print(datasets_samsung.describe(), datasets_hyundai.describe())
print(type(datasets_samsung), type(datasets_hyundai))

samsung_x = np.array(datasets_samsung.drop(['전일비', '종가'], axis=1))
samsung_y = np.array(datasets_samsung['종가'])
hyundai_x = np.array(datasets_hyundai.drop(['전일비', '종가'], axis=1))
hyundai_y = np.array(datasets_hyundai['종가'])

samsung_x = samsung_x[:180, :]
samsung_y = samsung_y[:180]
hyundai_x = hyundai_x[:180, :]
hyundai_y = hyundai_y[:180]

print(samsung_x.shape)

# samsung_x = samsung_x[:1200, :]
# samsung_y = samsung_y[:1200]
# hyundai_x = hyundai_x[:1200, :]
# hyundai_y = hyundai_y[:1200]

samsung_x = np.flip(samsung_x, axis=1)
samsung_y = np.flip(samsung_y)
hyundai_x = np.flip(hyundai_x, axis=1)
hyundai_y = np.flip(hyundai_y)

print(samsung_x.shape, samsung_y.shape)
print(hyundai_x.shape, hyundai_y.shape)

samsung_x = np.char.replace(samsung_x.astype(str), ',', '').astype(np.float64)
samsung_y = np.char.replace(samsung_y.astype(str), ',', '').astype(np.float64)
hyundai_x = np.char.replace(hyundai_x.astype(str), ',', '').astype(np.float64)
hyundai_y = np.char.replace(hyundai_y.astype(str), ',', '').astype(np.float64)

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, hyundai_x_train, hyundai_x_test, hyundai_y_train, hyundai_y_test = train_test_split(samsung_x, samsung_y, hyundai_x, hyundai_y,
                                                                                                                                                      train_size=0.7, shuffle=False)

scaler = MinMaxScaler()
samsung_x_train = scaler.fit_transform(samsung_x_train)
samsung_x_test= scaler.transform(samsung_x_test)
hyundai_x_train = scaler.transform(hyundai_x_train)
hyundai_x_test = scaler.transform(hyundai_x_test)

timesteps = 20

samsung_x_train_split = split_x(samsung_x_train, timesteps)
samsung_x_test_split = split_x(samsung_x_test, timesteps)
hyundai_x_train_split = split_x(hyundai_x_train, timesteps)
hyundai_x_test_split = split_x(hyundai_x_test, timesteps)

samsung_y_train_split = samsung_y_train[timesteps:]
samsung_y_test_split = samsung_y_test[timesteps:]
hyundai_y_train_split = hyundai_y_train[timesteps:]
hyundai_y_test_split = hyundai_y_test[timesteps:]

print(samsung_x_train_split.shape)      # (820, 20, 14)
print(hyundai_x_train_split.shape)      # (820, 20, 14)

# 2. 모델구성
# 2.1 모델1
input1 = Input(shape=(20, 14))
dense1 = LSTM(100, activation='relu', name='samsung1')(input1)
dense2 = Dense(200, activation='relu', name='samsung2')(dense1)
dense3 = Dense(300, activation='relu', name='samsung3')(dense2)
output1 = Dense(110, activation='relu', name='samsung4')(dense3)

# 2.2 모델2
input2 = Input(shape=(20, 14))
dense11 = LSTM(100, name='huyndai1')(input2)
dense12 = Dense(100, name='huyndai2')(dense11)
dense13 = Dense(100, name='huyndai3')(dense12)
dense14 = Dense(100, name='huyndai4')(dense13)
output2 = Dense(110, name='output2')(dense14)

# 2.3 머지
merge1 = Concatenate(name='mg1')([output1, output2])
merge2 = Dense(200, activation='relu', name='mg2')(merge1)
merge3 = Dense(300, activation='relu', name='mg3')(merge2)
hidden_output = Dense(100, name='last')(merge3)

# 2.5 분기1
bungi1 = Dense(10, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(10, name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)

# 2.6 분기2
last_output2 = Dense(1, activation='linear', name='last2')(hidden_output)
model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
hist = model.fit([samsung_x_train_split, hyundai_x_train_split], [samsung_y_train_split, hyundai_y_train_split], epochs=100, batch_size=64, validation_split=0.2, callbacks=[es])

model.save(path_save + 'keras53_samsung2_lsh.h5')

# 4. 평가, 예측
loss = model.evaluate([samsung_x_test_split, hyundai_x_test_split], [samsung_y_test_split, hyundai_y_test_split])
print('loss : ', loss)

result = model.predict([samsung_x_test_split, hyundai_x_test_split])

r2_1 = r2_score(samsung_y_test_split, result[0])
r2_2 = r2_score(hyundai_y_test_split, result[0])

print('r2_1 : ', r2_1)
print('r2_2 : ', r2_2)

result = np.array(result)
print(result.shape)
# result = result.reshape(2, 10)
result = result.reshape(2, 35)
result = result.T
print(result.shape)

last_result = np.round(result[34, 0], 2)
# last_result = np.round(result[9, 0], 2)
print("내일의 종가는 : ", last_result)