from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]

y = to_categorical(y)
print(y.shape) # (1797, 10)

x = x.reshape(1797,32,2,1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, stratify=y, 
    )

print(x.shape) # (1797, 32, 2, 1)

#2 모델
model = Sequential()
model.add(Conv2D(6, (2,2), padding='same', input_shape=(32,2,1)))
model.add(Conv2D(2, (4,4), padding='same'))
model.add(Conv2D(2, (4,4), padding='same'))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(10, activation = 'softmax'))

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics='acc')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train,y_train, epochs=10, batch_size=200,
          validation_split=0.2, verbose=1, callbacks=[es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

acc = accuracy_score(y_test_acc,y_predict)
print("acc : ", acc)