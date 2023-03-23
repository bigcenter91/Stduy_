from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, SimpleRNN
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import r2_score


# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항
print(train_csv.shape, test_csv.shape) # (1460, 80) (1459, 79)

print(train_csv.columns, test_csv.columns)

# 1.3 결측지
print(train_csv.isnull().sum())
print(train_csv.shape) # (1460, 80)


# 1.4 라벨인코딩
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
        
print(len(train_csv.columns))
print(train_csv.info())
train_csv=train_csv.dropna()
print(train_csv.shape) # (1121, 80)


# 1.5 x, y 분리
x = train_csv.drop(['SalePrice'], axis=1)
y = train_csv['SalePrice']

print(x.shape) # (1121, 79)
x = np.array(x)
x = x.reshape(-1, 79, 1)

# 1.6 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=777, shuffle=True)

#1.7 스케일링
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#2. 모델 구성
model = Sequential()

model.add(SimpleRNN(100, input_shape=(79, 1)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=200, verbose=1, 
                   mode='min', restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=2000, batch_size=79, verbose=1, 
                 validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)