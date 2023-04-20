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
    for i in range(len(dt)-st):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)

#1. 데이터

path = "d:/study_data/_data/project_p/"
save_path = "d:/study_data/_save/project_p/"

weather_g = pd.read_csv(path + 'OBS_212208_광주날씨.csv', index_col=0, encoding='cp949')
weather_j = pd.read_csv(path + 'OBS_212208_전주날씨.csv', index_col=0, encoding='cp949')
weather_m = pd.read_csv(path + 'OBS_212208_목포날씨.csv', index_col=0, encoding='cp949')
rice_k = pd.read_csv(path + '2122_미곡생산량_백미.csv', index_col=0, encoding='cp949')

print(weather_g)
print(weather_g.shape) # (62, 11)

print(weather_g.columns)
# Index(['평균기온(°C)', '최저기온(°C)', '최고기온(°C)', '10분 최다 강수량(mm)', '1시간 최다강수량(mm)',
#        '일강수량(mm)', '최대 순간 풍속(m/s)', '최대 풍속(m/s)', '평균 풍속(m/s)', '평균 이슬점온도(°C)',
#        '평균 상대습도(%)'],
#       dtype='object')

print(weather_g.info())
print(weather_g.describe())
print(type(weather_g)) # <class 'pandas.core.frame.DataFrame'>

wg_x = np.array(weather_g.drop(['일강수량(mm)','최대 순간 풍속(m/s)'], axis=1))
wg_y = np.array(weather_g['일강수량(mm)'])
wj_x = np.array(weather_j.drop(['일강수량(mm)','최대 순간 풍속(m/s)'], axis=1)) 
wj_y = np.array(weather_j['일강수량(mm)']) 



print(wg_x)
print(wg_x.shape, wg_y.shape) # (62, 9) (62,)
wg_x_train, wg_x_test, wg_y_train, wg_y_test, wj_x_train, wj_x_test, wj_y_train, wj_y_test = train_test_split(
    wg_x, wg_y, wj_x, wj_y, train_size=0.7, shuffle=False
)

print(wg_x_train.shape) # (43, 9)

scaler = MinMaxScaler()
wg_x_train = scaler.fit_transform(wg_x_train)
wg_x_test = scaler.transform(wg_x_test)
wj_x_train = scaler.transform(wj_x_train)
wj_x_test = scaler.transform(wj_x_test)

timesteps = 10

wg_x_train_split = split_x(wg_x_train, timesteps)
wg_x_test_split = split_x(wg_x_test, timesteps)
wj_x_train_split = split_x(wg_x_train, timesteps)
wj_x_test_split = split_x(wg_x_test, timesteps)

wg_y_train_split = wg_y_train[timesteps:]
wg_y_test_split = wg_y_test[timesteps:]
wj_y_train_split = wj_y_train[timesteps:]
wj_y_test_split = wj_y_test[timesteps:]

print(wg_x_train_split.shape) # (33, 10, 9)
print(wj_x_train_split.shape) # (33, 10, 9)


#2. 모델구성
#2.1 모델1
input1 = Input(shape=(timesteps, 9))
dense1 = LSTM(100, activation='relu', name='wg1')(input1)
flat = Flatten()(dense1)
dense2 = Dense(300, activation='relu', name='wg2')(flat)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(50, activation='relu', name='wg3')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(300, activation='relu', name='wg4')(drop2)
drop3 = Dropout(0.3)(dense4)
dense5 = Dense(200, activation='relu', name='wg5')(drop3)
drop4 = Dropout(0.3)(dense5)
output1 = Dense(33, activation='relu', name='wg6')(drop4)

#2.2 모델2
input2 = Input(shape=(timesteps, 9))
dense11 = LSTM(100, activation='relu', name='wj1')(input2)
flat = Flatten()(dense11)
dense12 = Dense(200, activation='relu', name='wj2')(flat)
drop1 = Dropout(0.3)(dense12)
dense13 = Dense(100, activation='relu', name='wj3')(drop1)
drop2 = Dropout(0.3)(dense13)
dense14 = Dense(300, activation='relu', name='wj4')(drop2)
output2 = Dense(33, name='output2')(dense14)

#2.3 머지
merge1 = Concatenate(name='mg1')([output1, output2])
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(20, activation='relu', name='mg3')(merge2)
hidden_output = Dense(33, name='last')(merge3)

#2.4 분기1
bungi1 = Dense(100, activation='selu', name='bg1')(hidden_output)
bungi2 = Dense(20, name='bg2')(bungi1)
last_output1 = Dense(1, name='last1')(bungi2)

#2.5 분기2
last_output2 = Dense(1, activation='linear', name='last2')(hidden_output)
model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)

hist = model.fit([wg_x_train_split, wj_x_train_split], [wg_y_train_split, wj_y_train_split], 
                 epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])

# model.save(path_save + 'keras53_wg2_kdj.h5')

#4. 평가, 예측
loss = model.evaluate([wg_x_test_split, wj_x_test_split], [wg_y_test_split, wj_y_test_split])
print('loss : ', loss)

wg_x_predict = wg_x_test[-timesteps:]
# print(wg_x_predict.shape)

wg_x_predict = wg_x_predict.reshape(1, timesteps, 9)

# print(wg_x_predict.shape) # (1, 10, 9)
wj_x_predict = wj_x_test[-timesteps:]
wj_x_predict = wj_x_predict.reshape(1, timesteps, 9)

predict_result = model.predict([wg_x_predict, wj_x_predict])

print("광주의 강수량은 : ", np.round(predict_result[0], 2))