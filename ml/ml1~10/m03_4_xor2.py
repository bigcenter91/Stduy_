# and 두개를 곱하는것, or 두개를 더하는것, xor : 두 값이 다를 경우 1반환(두값이 같으면 0, 다르면 1)

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

#1.데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]  #1이 하나라도 있어도 1

#2. 모델
model = SVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)

results = model.score(x_data, y_data)
print("model.score : ", results) #자동으로 모델에서 acc나 r2score로 조정해줌.

acc = accuracy_score(y_data, y_predict)
print('accuracy_score :', acc)

# model.score :  1.0
# accuracy_score : 1.0