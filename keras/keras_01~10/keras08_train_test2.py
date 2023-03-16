import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1,])


#[실습] 넘파이 리스트의 슬라이싱 7:3으로 잘라라

x_train = x[:7] #컴퓨터 숫자는 0부터니까
#print(x_train) # [1 2 3 4 5 6 7]
x_test = x[7:]
#print(x_test) # [8 9 10]
y_train = y[:7]
#print(y_train)
y_test = y[7:]
#print(y_test)

print(x_train.shape, x_test.shape) #(7,) (3,)
print(y_train.shape, y_test.shape) #(7,) (3,)

#훈련을 시킬 때는 범위 내에 훈련을 시키되 평가데이터에 쓰면 안된다
#앞으로는 전체 범위에서 비율로 뽑아 낼 것이다


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([11])
print('11의 예측값 : ', result)

'''
train 문제점
평가 하는 부분에서 오차가 적을 수 있으나 예측 값까지 갔을 때 점점 오차가 커질 수 있다
범위 밖에서는 오차가 커진다(loss) 명확하게 판단이 어렵다
= 훈련 시킬 때 가급적 전체 범위에서 훈련하되 범위 내에서 평가를 한다
train과 test를 분리하되, 전체 값에서 n%(일부의 비율)로 Test값 혹은 Train값을 뽑아 내야한다
'''