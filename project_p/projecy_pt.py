import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler,Normalizer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
def split_x(dt, st):
    a = []
    for i in range(len(dt)-st):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a)


#1. 데이터

path = "d:/study_data/_data/project_p/"
save_path = "d:/study_data/_save/project_p/"

# 기상 데이터 불러오기
weather_g21 = pd.read_csv(path + '2108_광주날씨.csv', index_col=0, encoding='cp949')
weather_g22 = pd.read_csv(path + '2208_광주날씨.csv', index_col=0, encoding='cp949')
weather_j21 = pd.read_csv(path + '2108_전주날씨.csv', index_col=0, encoding='cp949')
weather_j22 = pd.read_csv(path + '2208_전주날씨.csv', index_col=0, encoding='cp949')
weather_m21 = pd.read_csv(path + '2108_목포날씨.csv', index_col=0, encoding='cp949')
weather_m22 = pd.read_csv(path + '2208_목포날씨.csv', index_col=0, encoding='cp949')

print(weather_g21.columns)

# ['평균기온(°C)', '최저기온(°C)', '최고기온(°C)', '10분 최다 강수량(mm)', '1시간 최다강수량(mm)',
#        '일강수량(mm)', '최대 순간 풍속(m/s)', '최대 풍속(m/s)', '평균 풍속(m/s)', '평균 이슬점온도(°C)',
#        '평균 상대습도(%)'],
#       dtype='object')


# rice_21 = pd.read_csv(path + '2021_미곡생산량_백미.csv', index_col=0, encoding='cp949')
# rice_22 = pd.read_csv(path + '2022_미곡생산량_백미.csv', index_col=0, encoding='cp949')

weather_g = pd.concat([weather_g21, weather_g22])
weather_j = pd.concat([weather_j21, weather_j22])
weather_m = pd.concat([weather_m21, weather_m22])
# rice_k = pd.concat([rice_21, rice_22])

# weather21 = pd.concat([weather_g21, weather_j21, weather_m21])
# weather22 = pd.concat([weather_g22, weather_j22, weather_m22])

weather_g.index = pd.to_datetime(weather_g.index)
weather_j.index = pd.to_datetime(weather_j.index)
weather_m.index = pd.to_datetime(weather_m.index)
# rice_k.index = pd.to_datetime(rice_k.index)

print(weather_g)
print(weather_g.shape) # (62, 12)


# 1. 데이터 전처리
# 1) 결측치 처리
weather_g = weather_g.fillna(0)
weather_j = weather_j.fillna(0)
weather_m = weather_m.fillna(0)
# rice_k = rice_k.fillna(0)


# 2) MinMaxScaler를 이용한 데이터 스케일링
scaler = MinMaxScaler()
weather_g = scaler.fit_transform(weather_g)
weather_j = scaler.fit_transform(weather_j)
weather_m = scaler.fit_transform(weather_m)
# rice_k = scaler.fit_transform(rice_k)

# 3) 데이터셋 생성
# 2021, 2022년 데이터를 합쳐서 1년치 데이터로 생성
# 각 도시별로 각 일자의 데이터를 합쳐서 feature로 사용
# 각 도시의 일자별 강수량을 label로 사용
x = np.concatenate((weather_g, weather_j, weather_m), axis=1)
y = x[:, 4] # 4번째 feature는 강수량
x = np.delete(x, 4, axis=1) # 강수량 feature 삭제
x = split_x(x, 31) # 31일치 데이터로 1개의 데이터 생성
y = y[31:] # 31일치 이전의 강수량 데이터 삭제
y = y.reshape(-1, 1) # 강수량 데이터 shape 변경

# 4) Train, Test set split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# 2. 모델 구성
input1 = Input(shape=(31, x_train.shape[2]))
lstm1 = LSTM(64, activation='relu')(input1)
dense1 = Dense(32, activation='relu')(lstm1)
dense2 = Dense(16, activation='relu')(dense1)
output1 = Dense(1)(dense2)

model = Model(inputs=input1, outputs=output1)

# 3. 모델 학습 및 평가
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)

start_time = datetime.datetime.now()
hist = model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=1, validation_split=0.2, callbacks=[es])
end_time = datetime.datetime.now()

# 4. 학습 결과 시각화
plt.figure(figsize=(12, 8))
plt.plot(hist.history['loss'], marker='.', c='blue', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='red', label='val_loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()

# # 5. 모델 평가
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print('acc : ', acc)
# print("R2_SCORE: ", r2_score(y_test, y_predict))