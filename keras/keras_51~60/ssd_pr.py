from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, Embedding
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', ' 참 재밋네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요']

# 긍정인지 부정인지 맞춰봐!!!

x_predict = ['나는 성호가 정말 재미없다 너무 정말']

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])
# 긍정 1, 부정 0

data = docs + x_predict
print(data)

token = Tokenizer()
token.fit_on_texts(data)
print(token.word_index)
print(token.word_counts)

x = token.texts_to_sequences(data)
print(x)
# [[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]

# [[1, 7], [2, 8], [2, 3, 9, 10], [11, 12, 13], [14, 15, 16, 17, 18], [19], [20], [21, 22], [23, 24], [25], [1, 4], [2, 26], [5, 3, 27, 28], [5, 29], [30, 31, 6, 4, 1, 6]]

pad_x = pad_sequences(x, padding='pre', maxlen=5)
print(pad_x.shape)      # (15, 5)
pad_x_train = pad_x[:14, :]
pad_x_pred = pad_x[14, :]
pad_x_train = pad_x_train.reshape(pad_x_train.shape[0], pad_x_train.shape[1], 1)
pad_x_pred = pad_x_pred.reshape(1, 5, 1)

word_index = len(token.word_index)
print("단어사전의 갯수 : ", word_index)

# 2. 모델
model = Sequential()
model.add(Embedding(word_index, 32,'uniform',None,None,None,False,5))
# model.add(Embedding(28, 32, 5))           # error 
# model.add(Embedding(input_dim=28, output_dim=33, input_length=5))
# model.add(Reshape(target_shape=(5, 1), input_shape=(5,)))
# model.add(Dense(32, input_shape=(5,)))
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(pad_x_train, labels, epochs=100, batch_size=16)

# 4. 평가, 예측
acc = model.evaluate(pad_x_train, labels)[1]
print('acc : ', acc)

y_pred = np.round(model.predict(pad_x_pred))
print(y_pred)


# print(pad_x[0])
# x_pred= pad_x[0].reshape(1, 5, 1)
# y_pred_1 = np.round(model.predict(x_pred))
# print(y_pred_1)