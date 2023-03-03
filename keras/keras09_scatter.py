from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x,y, #train_size대로 나뉜다
        train_size=0.7, shuffle=True, random_state=1234)


#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x) #y=wxb인데 w가 곱해지겠지 / 두 차이를 보기 위해서
# 시각화를 한다고 표현
import matplotlib.pyplot as plt
plt.scatter(x, y )
#plt.scatter(x, y_predict)
plt.plot(x, y_predict, color='red')
plt.show()



                                                    
