from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=10, activation = 'sigmoid'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=5,
                 validation_split=0.2, 
                 verbose=1) # 훈련을 하는 과정은 fit에 있다

print(hist.history)

import matplotlib.pyplot as plt
plt.plot(hist.history['loss']) # 순서대로 있을 땐 x를 명시하지 않아도 된다
plt.show()