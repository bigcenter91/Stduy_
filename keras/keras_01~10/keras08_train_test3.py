import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법
#힌트 사이킷런

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
x, y, 
train_size=0.6,
test_size=0.3, # 둘 중 하나만 써도 된다
random_state=1234, 
shuffle=True,
#랜덤 시드라는게 있고 정의되어있는 표가 있다 랜덤시드로 고정을 시켜놓고 돌린다
#데이터 값이 좋아야한다
)

print(x_train)
print(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print("11의 예측 값 : ",result)

#The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.
#1.1이라는게 합이 '1'이 넘었다라는 뜻이다