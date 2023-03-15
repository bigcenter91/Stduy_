# 데이콘 따릉이 문제풀이

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd #numpy만큼 많이 쓴다



#1. 데이터
path = './_data/dacon_wine/' # . 하나가 현재 폴더
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0) #0번째는 index 컬럼이야
#train_csv = pd.read_csv('./_data/ddarung/train.csv')

print(train_csv)
print(train_csv.shape) # (5497, 13)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)

print(test_csv)
print(test_csv.shape) #(1000, 12)

# white는 0 red는 1로 바꿔줄 수 있겠지

from sklearn.preprocessing import LabelEncoder, RobustScaler #모델링하기 전에 정제하는거지 = 전처리
le = LabelEncoder() #스케일러랑 친구일거 같지않나?
le.fit(train_csv['type']) # 와인색이 어디 있어? 
aaa = le.transform(train_csv['type'])
print(aaa)
print(type(aaa)) # <class 'numpy.ndarray'>
print(aaa.shape) # (5497, )

print(np.unique(aaa, return_counts=True)) # (array([0, 1]), array([1338, 4159], dtype=int64))

train_csv['type'] = aaa
print(train_csv) # type 0,1로 변경된거 확인


test_csv['type'] = le.transform(test_csv['type'])
print(le.transform(['red', 'white'])) # [0 1] 순서대로 안됐으면 어떡하죠?
print(le.transform(['white', 'red'])) # [1 0] 거꾸로도 찍어봐라
#수치로 되어있지 않고 스트링으로 되어있는 형태를 수치로 바꿔준다

# 라벨인코더 정의하고 어떤 애들을 핏해주고 대상이 정해졌다면 그거에 맞춰서 트랜스폼 
# 정의 > 핏하고 > 트랜스폼

# scaler = RobustScaler()
# scaler.fit(x_train)



#==============================================================
'''
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

x = train_csv.drop(['count'], axis=1) # ['']] 2개 이상은 리스트
print(x)
y = train_csv['count']
print(y) 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=777
)
 
print(x_train.shape, x_test.shape) # (1021, 9) (438, 9) > (929, 9) (399, 9)
print(y_train, y_test.shape) #(1021, ) (438, ) > (929, 399)

#############################train_csv 데이터에서 x와 y를 분리 #############################

model = Sequential() # keras에서는 Sequential 모델하고 나중에 배울 함수모델이 있다
model.add(Dense(10, input_dim=9))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
'''