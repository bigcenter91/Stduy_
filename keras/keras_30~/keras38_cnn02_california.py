from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

print(x.shape) # (20640, 8)

x = x.reshape(20640, 4, 2, 1)

print(x.shape) # (20640, 4, 2, 1)
print(y.shape) # (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,)

#2. 모델 구성
model = Sequential()

model.add(Conv2D(2, (2,1), 
                #   padding='same',
                 input_shape=(4,2,1))) # 출력 : 
                                
model.add(Conv2D(filters=4, 
                #   padding='same',
                 kernel_size=(2,1),
                 activation='relu')) 

model.add(Conv2D(10, (2,1),)) 
                            
model.add(Flatten())         
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))      
model.add(Dense(1, activation='linear'))
model.summary()

print(np.unique(y_train, return_counts=True))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='acc')
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