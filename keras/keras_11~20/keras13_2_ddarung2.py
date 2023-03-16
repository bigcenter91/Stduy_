# 데이콘 따릉이 문제풀이

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd #numpy만큼 많이 쓴다

#1. 데이터
path = './_data/ddarung/' # . 하나가 현재 폴더
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0) #0번째는 index 컬럼이야
#train_csv = pd.read_csv('./_data/ddarung/train.csv')

print(train_csv)
print(train_csv.shape) # (1459, 10) # index 혹은 id 데이터가 아니다 그냥 번호다 y=ax+b id를 넣고 연산할 필요가 없다
#pandas는 인덱스와 컬럼명이 따라다닌다 index는 따로 연산하지 않는다

test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)

print(test_csv)
print(test_csv.shape) #(715, 9)

#==============================================================

print(train_csv.columns)
#Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#     dtype='object')

print(train_csv.info())
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64



print(train_csv.describe())


############################# 결측치 처리 #############################
#특정 값을 넣어도 되지만 삭제할 수도 있다
#결축치 처리 1. 제거
#print(train_csv.isnull())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()  ### 결측치 제거
print(train_csv.isnull().sum())
print(train_csv.info()) # info 각 컬럼의 정보 확인
print(train_csv.shape) #(1328, 10)




#############################train_csv 데이터에서 x와 y를 분리 #############################

# 데이터 받았을 때 


#2. 모델 구성
print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>

x = train_csv.drop(['count'], axis=1) # [''] 2개 이상은 리스트
print(x)
y = train_csv['count']
print(y) 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=2000
)
 
print(x_train.shape, x_test.shape) # (1021, 9) (438, 9) > (929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(1021, ) (438, ) > (929, 399)

#############################train_csv 데이터에서 x와 y를 분리 #############################

model = Sequential() # keras에서는 Sequential 모델하고 나중에 배울 함수모델이 있다
model.add(Dense(32, input_dim=9))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=6,
          verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
# nan 뜨는 이유 결측치가 너무 많아서
# nan값 결측치 처리

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict): #재사용 할 때 함수를 쓴다/ RMSE라는 함수를 정의 할거야
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt 루트를 씌우는 놈이다
rmse = RMSE(y_test, y_predict) # 정의한 RMSE 사용
print("RMSE : ", rmse)


##### submission.csv를 만들어봅시다 #####
#print(test_csv.isnull().sum())  # 여기도 결측치가 있다
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path_save + 'submit_0303.csv')