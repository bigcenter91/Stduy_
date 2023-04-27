import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

#1 데이터
path1 = 'c:/study_data/_data/dust/TRAIN/'
path2 = 'c:/study_data/_data/dust/TRAIN_AWS/'
path3= 'c:/study_data/_data/dust/TEST_INPUT/'
path4 = 'c:/study_data/_data/dust/TEST_AWS/'
save_path = 'c:/study_data/_data/dust/'

csv_files = [f for f in os.listdir(path1) if f.endswith('.csv')]

dataframes1 = []

for file in csv_files:
    file_path = os.path.join(path1, file)
    df = pd.read_csv(file_path)
    dataframes1.append(df)

csv_files = [f for f in os.listdir(path2) if f.endswith('.csv')]

dataframes2 = []

for file in csv_files:
    file_path = os.path.join(path2, file)
    df = pd.read_csv(file_path)
    dataframes2.append(df)
    
csv_files = [f for f in os.listdir(path3) if f.endswith('.csv')]

dataframes3 = []

for file in csv_files:
    file_path = os.path.join(path3, file)
    df = pd.read_csv(file_path)
    dataframes1.append(df)

csv_files = [f for f in os.listdir(path4) if f.endswith('.csv')]

dataframes4 = []

for file in csv_files:
    file_path = os.path.join(path4, file)
    df = pd.read_csv(file_path)
    dataframes2.append(df)
    
dataframes1.extend(dataframes3)
dataframes2.extend(dataframes4)

# 데이터프레임 연결
all_data = pd.concat(dataframes1 + dataframes2, axis=0)

# 필요한 열만 선택
df = all_data[['PM2.5']]

# 데이터 전처리 및 정규화
scaler = MinMaxScaler()
df['PM2.5'] = scaler.fit_transform(df[['PM2.5']])

# 인덱스 재설정
df.reset_index(drop=True, inplace=True)

# 시계열 데이터를 X와 y로 분할
sequence_length = 72  # 3일 분량의 데이터 (24시간 x 3일 = 72시간)

X = []
y = []

for i in range(len(df) - sequence_length - 1):
    X.append(df['PM2.5'][i:i + sequence_length].values)
    y.append(df['PM2.5'][i + sequence_length])

X = np.array(X)
y = np.array(y)

# 데이터를 학습 및 테스트 셋으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

# LSTM 입력을 위해 3차원 데이터로 변환
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(30))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# 모델 학습
model.fit(X_train, y_train, epochs=1, batch_size=10000, validation_split=0.2, verbose=1)

# 예측
y_pred = model.predict(X_test)

# 예측값을 원래 스케일로 변환
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)

print("예측 결과:")
for i in range(5):
    print(f"실제 PM2.5: {y_test[i][0]:.2f}, 예측 PM2.5: {y_pred[i][0]:.2f}")

# 예측 결과를 데이터프레임으로 변환
result_df = pd.DataFrame({'Actual_PM2.5': y_test.flatten(), 'Predicted_PM2.5': y_pred.flatten()})

# 결과를 CSV 파일로 저장
result_df.to_csv('dust_pm25_predictions.csv', index=False)

print("예측 결과가 'dust_pm25_predictions.csv' 파일로 저장되었습니다.")