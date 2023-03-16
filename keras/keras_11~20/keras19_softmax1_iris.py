import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_iris()
print(datasets.DESCR) #
print(datasets.feature_names) # 판다스(cl)


x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(150,4) / (150,)
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y)) #y의 라벨값 :  [0 1 2]

#################이지점에서 원핫을 해야겠지?#################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(150, 3)


#판다스 겟더미, 사이킷런 원핫인코더

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, #random_state=333#,
    test_size=0.2,
    stratify=y, #이론 상으로 써주는게 맞다 / 안넣어줄 때 값이 좋을 때가 있다

)

print(y_train)
print(np.unique(y_train, return_counts=True))

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=4))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) 

# 다중분류에서는 무조건 softmax 안바뀐다 / 출력값은 150, 3이된다 
#다중분류 문제는 마지막 레이아웃(out layer)에 y의 라벨 값의 노드를 적어준다
#마지막 히든 레이어 3은 y의 라벨 값의 갯수
# 0~1사이 한정 시키는 놈이다


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=2,
          validation_split=0.3,
          verbose=1,
          )

#4. 평가, 예측
#results로 한 이유는 값이 두개가 나오기 때문에
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)

# print(y_test.shape) # (30, 3)
# print(y_pred.shape) # (30, 3)
# # softmax를 해서 수치화된 데이터일거 아냐? (0.570041298866272식으로)

# print(y_test[:5])
# print(y_pred[:5])

# acc = accuracy_score(y_test, y_pred)
# print(acc)

y_test_acc = np.argmax(y_test, axis=1) # 각 행에 있는 열끼리 비교
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_acc : ', acc)

#=======================================
# #4. 평가, 예측
# results = model.evaluate(x_test, y_test)
# print('results : ', results)

# y_predict = model.predict(x_test)
# print(y_predict.shape) # (30, 3)

# from sklearn.metrics import r2_score, accuracy_score
# y_predict=np.argmax(y_predict, axis=1)

# print(y_predict.shape)

# print(y_test.shape)

# y_test = np.argmax(y_test, axis=1)
# print(y_test.shape)

# acc =accuracy_score (y_test, y_predict)

# print('acc : ', acc)

# 열을 기준으로 타고 내려감
# argmax 위치값의 순서를 반환해준다
# accuracy_score를 사용해서 스코어를 빼세요