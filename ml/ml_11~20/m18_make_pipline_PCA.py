# PCA 주성분 분석의 차원축소 기법인데
# python에서는 컬럼의 갯수를 축소

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

#1. 데이터
x, y = load_digits(return_X_y=True)
print(x.shape) # (1797, 64) digits : mnist의 축소형 sklearn의 모델은 2차원만 받는다
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
# array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

# 8개로 압축해볼까?
# pca = PCA(n_components=8)
# x = pca.fit_transform(x)
# print(x.shape) # (1797, 8)
# 삭제 개념이 아니라 압축 개념이다 오히려 성능이 더 좋아지는 경우도 있다
# 디폴트로 했을 땐 변환되고 그냥 값이 바뀐다 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True
)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = RandomForestClassifier()

# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
model = make_pipeline(PCA(n_components=8), StandardScaler(), SVC())
# 이 상황은 pca를 train_test_split 한다음에 pca가 진행된거다


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)