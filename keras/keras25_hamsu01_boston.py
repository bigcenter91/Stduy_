#이미지는 무조건 4차원_가로, 세로, 칼라, 장 수

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)


print(np.min(x), np.max(x)) # x의 최소값 / (0.0 711.0)
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.min(x), np.max(x)) # (0.0 1.0)
#훈련데이터만 정규화한다 0~1사이로
#테스트 데이터 정규화 하는데 훈련데이터를 잡고 훈련데이터의 비율에 맞춰서
#x_predict 미래를 알고싶은놈
#ex) 110-0 / 100-0 = 1.1 0은 훈련데이터에서 온 것
#분리한다음에 스케일링 한다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)

#scaler = MinMaxScaler()
#scaler = StandardScaler() # MinMaxScaler, StandardScaler 둘 중 하나 / 하나로 모아줘야하면 스탠다드, 그 반대는 민맥스
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x_test), np.max(x_test))


#2. 모델
model = Sequential()
model.add(Dense(30, input_shape=(13,), name='S1'))
model.add(Dense(20, name='S2'))
model.add(Dense(10, name='S3'))
model.add(Dense(1, name='S4'))
model.summary()



input1 = Input(shape=(13,), name='h1')
dense1 = Dense(30, name='h2')(input1)
dense2 = Dense(20, name='h3')(dense1)
dense3 = Dense(10, name='h4')(dense2)
output1 = Dense(1, name='h5')(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()

#시퀀셜 모델은 상단에서하고 함수형 모델에서는 마지막에 한다
#모델의 정의를 상단에서 하느냐 하단에서 하느냐

# 데이터가 3차원이면(시계열 데이터)
#(1000, 100, 1) >> input_shape= 100, 1 / 행빼고
# 데이터가 4차원이면(이미지 데이터)
# (60000, 32, 32, 3) >> input_shape=(32, 32, 3) / 제일 앞에가 데이터 갯수이고 행이다 그래서 행 무시, 열 우선
#앞으론 input_shape로 쓸거다
'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss )
'''