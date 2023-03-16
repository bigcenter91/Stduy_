from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=8, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu')) 
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=4,
                 validation_split=0.2, 
                 verbose=1)

print(hist.history)

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # 순서대로 있을 땐 x를 명시하지 않아도 된다
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='loss')
plt.title('california')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show() 