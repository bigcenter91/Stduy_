from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x,y, #train_size대로 나뉜다
        train_size=0.8, shuffle=True, random_state=1234)


#2. 모델 구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(35))
model.add(Dense(18))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=5)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

#R2 r2 결정계수

from sklearn.metrics import r2_score #predict 예측한 y값
r2 = r2_score(y_test, y_predict)

print('r2 스코어 : ', r2)

#r2 스코어 0.99로 맞춰봐라
#r2값이 -1가 나오면 안좋은 지표다 r2는 보조지표다 loss와 같이 판단한다


#train_size /random_state  변경해도 된다

                                                    
