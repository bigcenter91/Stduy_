import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1,])
# print(x)
# print(y)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) #x_test / y_test는 훈련에 포함되지 않는다
print('loss : ', loss)

result = model.predict([11])
print('11의 예측값 : ', result) # 범위 밖은 오차가 클 수 밖에 없다

# loss :  0.013200442306697369 // 11의 예측값 :  [[10.984116]] //  epochs=150, batch_size=2
# loss :  0.00042263665818609297 // 11의 예측값 :  [[11.000328]] // epochs=200, batch_size=3