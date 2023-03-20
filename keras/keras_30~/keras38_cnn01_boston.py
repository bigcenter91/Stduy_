from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from tensorflow.sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets['target']
x = x.reshape(506, 13, 1, 1)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
)

print(x_train.shape) # (404, 13)
print(x_test.shape) # (102, 13)
print(y_train.shape) # (404,)
print(y_test.shape) # (102, )

# x_train = x_train.reshape(404, 13, 1, 1)
# x_test = x_test.reshape(102, 13, 1, 1)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape) # (404, 51)
# print(y_test.shape) # (102, 51, )

# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test) 




#2. 모델 구성
model = Sequential()

model.add(Conv2D(3, (2,2), 
                  padding='same',
                 input_shape=(13,1,1))) # 출력 : 
                                
model.add(Conv2D(filters=4, 
                  padding='same',
                 kernel_size=(2,2),
                 activation='relu')) 

model.add(Conv2D(10, (2,2),
                 padding='same',)) 
                            
model.add(Flatten())         
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))      
model.add(Dense(1, activation='linear'))
model.summary()

print(np.unique(y_train, return_counts=True))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                    verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=200, batch_size=1000,
                 validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# import matplotlib.pyplot as plt
# plt.plot(hist.history['val_loss'], label='val_acc')
# plt.show()