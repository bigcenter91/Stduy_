from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print(x.shape) # (581012, 54)
print(y.shape) # (581012, )

print(np.unique(y)) # [1 2 3 4 5 6 7]

y = to_categorical(y)
print(y.shape) # (581012, 8)


x = x.reshape(581012, 9, 6, 1)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.7, shuffle=True, random_state=123)

print(x.shape,y.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(3, (2,2), padding='same', input_shape=(9,6,1))) 
                                
model.add(Conv2D(4, (2,2), padding='same',
                 activation='relu')) 

model.add(Conv2D(10, (2,2), padding='same',)) 
                            
model.add(Flatten())         
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))      
model.add(Dense(7, activation='softmax'))


#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics='acc')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                   verbose=1, restore_best_weights=True)

model.fit(x_train,y_train, epochs=10, batch_size=20,
          validation_split=0.2, verbose=1, callbacks=[es])

#4 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_acc : ', acc)
