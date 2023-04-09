from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)
# 단어의 빈도 수만큼 숫자가 나온다 10000이면 왠만큼 다 나오겠지?
# 임베딩의 인 풋 딤 10000개의 데이터
# input_legth 최대길이 혹 모르면 안넣어도 된다


print(x_train)
print(y_train) # [ 3  4  3 ... 25  3 25]
print(x_train.shape, y_train.shape) # (8982,) (8982,)
print(x_test.shape, y_test.shape) # (2246,) (2246,)

print(len(x_train[0]), len(x_train[1])) # 87, 56 // numpy 안의 리스트 형식으로 되있어서 나온다
print(np.unique(y_train)) # 46개 클래스파일

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/ len(x_train)) # 145.84948045522017

# 길이를 일일히 다 재서 끝까지


# 전처리

from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating='pre') # 버리는 놈 truncating 
#padding은 100개 미만의 것의 앞부분을 0으로 채우겠다

print(x_train.shape) # (8982, 100)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 전처리

pad_x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
pad_x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

# softmax 46, embedding input_dim=10000, output_dim=마음대로, input_length=max(len)
pad_x_train = pad_x_train.reshape(pad_x_train.shape[0], pad_x_train.shape[1], 1)
pad_x_test = pad_x_test.reshape(pad_x_test.shape[0], pad_x_test.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(Embedding(10000, 32, input_shape=(100,)))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(46, activation='softmax'))



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x_train, y_train, epochs=100, batch_size=256)

#4. 평가, 예측
acc = model.evaluate(pad_x_test, y_test)[1]
print('acc : ', acc)