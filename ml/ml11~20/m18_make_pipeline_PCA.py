# 컬럼의 개수를 축소할 수 있음. (컬럼을 압축해버림.(조절가능))
#소문자 함수
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
#1.데이터
x,y = load_digits(return_X_y=True) 
print(x.shape) #(1797, 64) #사이킷모델은 2차원밖에 못받음.
print(y.shape) #(1797,)
print(np.unique(y, return_counts= True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
#array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

# pca = PCA(n_components= 8) #몇개의 컬럼으로 압축할꺼냐?
# x = pca.fit_transform(x)
print(x.shape) #(1797, 8) #압축이라 성능이 더 좋아질수도 있음.
#디폴트는 변환이 됐지만, 값은 약간씩다름.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

#2. 모델
model = make_pipeline(PCA(n_components= 8),StandardScaler(),SVC()) 

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)