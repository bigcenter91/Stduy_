from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder


text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 배환희다. 멋있다. 또 또 얘기해부아'
# 문장을 토큰(어절) 별로 잘라내는 개념이다
# 알아듣게 하기 위해서 수치화를 해야한다 // 수치를 지정 해줘야한다
# 잘라낸다음 어절 단위로 수치화 해준다

token = Tokenizer()
token.fit_on_texts([text1, text2])
print(token.word_index) 
# {'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는': 6, '밥을': 7, '엄청': 8, '먹었다': 9, '지구용사': 10, '배환희다': 11, '멋있다': 12, '얘기해부아': 13}

# 가장 많은 놈이 앞에 인덱싱을 준거다 ex 마구, 매우 그 다음 부턴 순서대로

print(token.word_counts)
# OrderedDict([('나는', 2), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1), ('지구용사', 1), ('배환희다', 1), ('멋있다', 1), ('또', 2), ('얘 
# 기해부아', 1)])
# token.word_counts: 단어 갯수

# 숫자로 바꿔주는 작업
x = token.texts_to_sequences([text1, text2])
print(x)
# [[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]
print(type(x)) # <class 'list'>
# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] // 1행, 11열 : 작은거부터 세는거지?
# 그냥 연산하면 숫자 높은게 가치가 높다고 판단하겠지? 그렇게 판단할 수 있으면 안되지?
# 그래서 원핫해줘야한다

x = x[0] + x[1]
print(x)

###### 1. to_categorical ######
# 11개의 행이 나오겠지 일단
# from tensorflow.keras.utils import to_categorical


# x = to_categorical(x)
# print(x)
# print(x.shape) # (18, 14) to_categorical 때문에 앞에 0이 붙어서 18, 14
# 필요없는 0번째 1열이 생겼지?

###### 2. get_dummies ######   // 1차원으로 받아들여
import pandas as pd

# x = pd.get_dummies(np.array(x).reshape(11,)) # 1차원으로 바꿔줘야한다
x = pd.get_dummies(np.array(x).ravel())
x = np.array(x)
print(x.shape) # (11, 8)

# list라서 오류발생 list에서 numpy로 바꿔줘야한다
# 1차원이여서 발생하는 오류
# get_dummies는 1차원만 받으니까 바꿔줘

###### 3. 사이킷런 원핫 ######  // 2차원으로 받아들여야한다 // 그래서 ravel 안먹힌다
# from sklearn.preprocessing import OneHotEncoder
# # 알아서 맹그러

# enc = OneHotEncoder()
# x = enc.fit_transform(np.array(x).reshape(-1,1)).toarray()

# print(x)

# print(x.shape)

# 결론은 편한거 쓰면 된다
