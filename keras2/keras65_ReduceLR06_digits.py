import numpy as np


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (1797, 64) (1797,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=524, shuffle=True,
)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
from tensorflow.keras.optimizers import Adam
learning_rate = 0.00005
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer)


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1,
                        factor=0.5)
model.fit(x_train, y_train, epochs=200, batch_size=8, verbose=1, validation_split=0.2,
          callbacks=[es, rlr])


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss :", results)


