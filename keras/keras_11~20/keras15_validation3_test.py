from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17)) # 16개의 데이터죠?
y = np.array(range(1,17)) # range = 16 스칼라 1 벡터(특성이 하나) 1 디맨션

# x_val = np.array([14,15,16]) #16개 데이터 중에 3개도 되고, 13개 중 3개
# y_val = np.array([14,15,16])
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])

#실습 :: 잘라봐!
#train_test_split로만 잘라라
#10:3:3

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=13/16, # 13 : 13 : 3 : 3이 된다
                                                                            # transize의 앞에 숫자가 train으로 간다 - 13/16
                                                random_state=1, shuffle=True)
#print(x_train, y_train, x_test, y_test)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=10/13,
                                                random_state=1, shuffle=True) # 위에서 처리됐으니까 10/13이된거다

print(x_train, y_train, x_val, y_val, x_test, y_test)


#2. 모델
model= Sequential()
model.add(Dense(5, activation='linear', input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) #특성, 피쳐 하나

#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val)) #train data가 제일크긴해야한다

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)
