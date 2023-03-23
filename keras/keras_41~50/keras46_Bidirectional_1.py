import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9]]) # 8, 9, 10 y가 있기 때문에

y=np.array([4, 5, 6, 7 ,8 ,9 ,10])

x = x.reshape(7, 3, 1)

#2. 모델 구성

model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(3,1)))
model.add(LSTM(10, return_sequences=True))
model.add(Bidirectional(GRU(10)))

model.summary()
#bidirectional은 혼자 사용 안된다 / RNN을 랩핑한다는 뜻
# 기존 RNN보다 120개 늘어남
# LSTM은 960
# 한마디로 rnn을 랩핑하면 더블이다
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  bidirectional (Bidirectiona  (None, 20)               240
#  l)

#  dense (Dense)               (None, 1)                 21

# =================================================================