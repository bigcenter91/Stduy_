import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)

#print(train_csv)
#print(train_csv.shape) # (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col=0)

#print(test_csv)
#print(test_csv.shape) # (6493, 8) /index를 제거할 필요없다

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
# 캐쥬얼과 레지스터 컬럼을 삭제하는게 편하다 / 그럼 x의 input_dim은 8
print(test_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed'],
#       dtype='object')


print(train_csv.info())
#  0   season      10886 non-null  int64
#  1   holiday     10886 non-null  int64
#  2   workingday  10886 non-null  int64
#  3   weather     10886 non-null  int64
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64
#  9   registered  10886 non-null  int64
#  10  count       10886 non-null  int64


print(train_csv.describe())


######################결측치 처리#####################
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape) #(10886, 11)



#####################train_csv 데이터에서 x와 y를 분리 #####################

print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>
x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv ['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=123)

print(x_train.shape, x_test.shape) # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) # (7620,) (3266,)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=8, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu')) #리니어는 디포트로 되어있는거다 그냥 대각선
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

#음수 값으로 나오는 경우가 있다
#그럴 때 활성화 함수를 써준다 Activation 다음 레이어로 전달하는 값을 한정시킨다
#Relu 0이상의 값은 양수가 되고 0이하의 값은 0으로 된다 hidden layer에 많이 쓴다

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=4,
          validation_split=0.2, verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


