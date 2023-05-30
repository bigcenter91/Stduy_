#그물망 (파라미터 전체를 다하겠다.)
#대부분 fit이나 model에서 파라미터를 정리함.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state=953, stratify=y
)

gamma = [0.001, 0.01, 0.1, 1, 10, 100]
C = [ 0.001, 0.01, 0.1, 1, 10, 100]
#-> 파라미터

max_score = 0

for i in gamma:
    for j in C:
        #2. 모델
        model = SVC(gamma = i, C = j)

        #3. 컴파일, 훈련
        model.fit(x_train,y_train)

        #4. 평가, 예측
        score = model.score(x_test, y_test)
        
        if max_score < score :
            max_score = score #earlystopping이랑 비슷함(계속 최댓값만 저장.)
            best_parameters ={'gamma': i, 'C' : j}

print("최고점수 : ", max_score)
print("최고의 매개변수 : ", best_parameters)

# 최고점수 :  1.0
# 최고의 매개변수 :  {'gamma': 0.1, 'C': 0.001}
#여기서는 1이 여러개 나오므로 먼저 나온 순서가 나옴.

# 분류 모델로 예시로 SVM 사용
# svc = SVC()

# # 탐색할 매개변수 그리드 설정
# param_grid = {'kernel': ['linear', 'rbf', 'sigmoid'], 'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

# # GridSearchCV 객체 생성
# grid_search = GridSearchCV(svc, param_grid)

# # 데이터셋을 불러와서 분할 후 학습

# grid_search.fit(x_train, y_train)

# # 최적 매개변수와 테스트 정확도 출력
# print("최적 매개변수:", grid_search.best_params_)
# print("테스트 세트 점수:", grid_search.score(x_test, y_test))