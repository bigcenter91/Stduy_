from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Conv1D, Reshape, Embedding

#1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', ' 참 재밋네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요']

# 이미 리스트 형태

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)
# [[2, 5], [1, 6], [1, 3, 7, 8, 9, 10], [11, 12, 13], [14], [15], [16, 17], [18, 19], [20], [2, 21], [1, 22], [4, 3, 23, 24], [4, 25]]

# 0으로 채우는게 패딩이였지?
# 조선시대부터 자연어처리를 해왔다

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # pre 앞에서부터 0을 채운다
# 1, 3, 7, 8, 9, 10 // maxlen을 4로 주게 되면 5자리는 앞에가 잘린다
print(pad_x)
print(pad_x.shape) # (14, 5)
pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1])


word_size = len(token.word_index)
print("단어사전의 갯수: ", word_size) # 단어사전의 갯수:  28

# 인 풋 쉐입 5 // 아웃 1
# 자체가 시계열 데이터다



#2. 모델
model = Sequential()
model.add(Embedding(28, 32, input_length=5))
# model.add(Dense(32, input_shape=(5,)))
# model.add(Reshape(target_shape=(5, 1), input_shape=(5, ))) # reshape 따로 하지 않고 여기서 해줘도 된다
model.add(LSTM(32))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

model.fit(pad_x, labels, epochs=30, batch_size=8)

#4. 평가, 예측

acc = model.evaluate(pad_x, labels)[1]
print('acc :', acc)

# acc : 0.7857142686843872 Dense
# acc : 0.9285714030265808 LSTM
# acc : 0.9285714030265808 Embeding

# embeding 한마디로 효율적인 원핫이라고 볼 수 있다
# 좌표계의 데이터를 집어 넣었다고 생각해 그리고 텍스트에 아조 좋다