from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


####### 실습 ####### 맹그러봐

x_train = x_train.reshape(60000, 28, 28, 1) # (60000, 28, 14, 2)도 가능하다 곱하거나 나눠야한다 그렇게해서 원값이 나와야한다
                                            # shape와 크기는 같아야한다 
                                            # 행은 건들면 안된다 / 4차원 데이터
x_test = x_test.reshape(10000, 28, 28, 1)
#reshape 할 때 데이터 구조만 바뀌지 순서와 내용은 바뀌지 않는다 / transepose는 열과 행을 바뀐다 반대로 구조가 바뀌는거다


print(x_test.shape) # (10000, 28, 28, 1)
print(y_test.shape) # (10000, )

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) # (60000, 10)
print(y_test.shape) # (10000, 10)
# y가 N, 10이 되야겠지 그럼 y 원핫해야겠지


scaler = MinMaxScaler() #Minmax를 쓸려면 2차원으로 바꿔줘야한다
print(x_train.shape)
x_train = x_train.reshape(-1,1)
print (x_train.shape) # (47040000, 1) / 2차원
print (x_train.shape[0]) # 47040000

x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)



#2. 모델 구성
model = Sequential()

model.add(Conv2D(7, (2,2), 
                 padding='same',
                 input_shape=(28,28,1))) # 출력 : (N, 7, 7, 7) 
                                
model.add(Conv2D(filters=4, 
                 padding='same',
                 kernel_size=(3,3),
                 activation='relu')) 

model.add(Conv2D(10, (2,2))) 
                            
model.add(Flatten())         
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))      
model.add(Dense(10, activation='softmax'))
model.summary()


print(np.unique(y_train, return_counts=True))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                    verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=200, batch_size=1000,
                 validation_split=0.2, verbose=1, callbacks=[es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ', acc)

import matplotlib.pyplot as plt
plt.plot(hist.history['val_loss'], label='val_acc')
plt.show()


#이미지 데이터는 cnn이 좋다 보편적으로
# result :  [0.0828157439827919, 0.9764999747276306] // acc :  0.9765
# result :  [0.09381601959466934, 0.9753999710083008] // acc :  0.9754
# result :  [0.0904429629445076, 0.9735999703407288] // acc :  0.9736